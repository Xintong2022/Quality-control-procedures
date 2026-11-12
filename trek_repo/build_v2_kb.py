"""
build_v2_kb.py — TravelBench KB v2 cleanup script.

Applies all approved fixes from KB_CLEANUP_DESIGN.md:
  H1: hotel `about` rating denominator /5 -> /10  (39,596 records)
  H2: hotel `about` half-star X-star -> X.X-star  (12,902 records)
  H3: delete both Barcelona cities + 7 referencing queries
  H4: hotel `address` locale-aware regeneration  (39,396 after H3)
  A1: attraction `address` locale-aware regeneration  (55,827 after H3)
  A3: attraction `overview` open_hours full-string fix  (6 records)

Inputs  (current state of api/data/):
  hotel_data/, attraction_data/, car_data/, flight_data/, new_travelbench/merged_query.csv

Outputs (created under api/data/):
  hotel_data_v2/, attraction_data_v2/, car_data_v2/, flight_data_v2/
  new_travelbench/merged_query_v2.csv
  KB_AUDIT_v2.md  (audit report)

Run from project root:
  cd /Users/jinhuqi/Downloads/KDD2026.../KDD_travelbench
  python3 build_v2_kb.py
"""
import pandas as pd
import numpy as np
import re
import os
import math
import shutil
import glob
import json
import unicodedata
import ast
from collections import defaultdict

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
RNG_SEED = 42

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, 'api', 'data')

IN_HOTEL   = os.path.join(DATA, 'hotel_data')
IN_ATTR    = os.path.join(DATA, 'attraction_data')
IN_CAR     = os.path.join(DATA, 'car_data')
IN_FLIGHT  = os.path.join(DATA, 'flight_data')
IN_QUERY   = os.path.join(ROOT, 'new_travelbench', 'merged_query.csv')

OUT_V2     = os.path.join(DATA, 'v2')
OUT_HOTEL  = os.path.join(OUT_V2, 'hotel_data')
OUT_ATTR   = os.path.join(OUT_V2, 'attraction_data')
OUT_CAR    = os.path.join(OUT_V2, 'car_data')
OUT_FLIGHT = os.path.join(OUT_V2, 'flight_data')
OUT_QUERY  = os.path.join(OUT_V2, 'merged_query.csv')

AUDIT_PATH = os.path.join(OUT_V2, 'KB_AUDIT.md')

# Files to skip (Barcelona)
BARCELONA_FILES = {
    'Barcelona_Spain_hotel.csv', 'Barcelona_Venezuela_hotel.csv',
    'Barcelona_Spain_attraction.csv', 'Barcelona_Venezuela_attraction.csv',
    'Barcelona_Spain_rental_cars.csv', 'Barcelona_Venezuela_rental_cars.csv',
}

# .npy embedding prefixes to skip (Barcelona disambiguated names)
BARCELONA_NPY_PREFIXES = (
    'Barcelona_Spain_', 'Barcelona_Venezuela_',
)

# H3 deletes both Barcelonas because one name over two cities is exactly the confusion the
# benchmark is meant to measure the agent resolving, not to inflict on it. These are their
# airports: BCN (Spain), BLA (Venezuela).
BARCELONA_AIRPORTS = {'BCN', 'BLA'}

# ----------------------------------------------------------------------------
# Locale-aware address templates (Bug H4 / A1)
# ----------------------------------------------------------------------------
# Each country maps to (template, street_pool).
# Template uses placeholders {num}, {street}, {city}.
# Sub-district is intentionally omitted to avoid coord-address inconsistency.

# Generic neutral street name lists — designed to look locale-appropriate
# without referencing real businesses/landmarks.

STREETS_EN = [  # English-speaking generic
    'Maple Street', 'Oak Avenue', 'Pine Road', 'Cedar Lane', 'Birch Boulevard',
    'Elm Street', 'Park Avenue', 'Main Street', 'Central Avenue', 'River Road',
    'Lakeview Drive', 'Hillcrest Road', 'Sunset Boulevard', 'Highland Avenue',
    'Forest Lane', 'Meadow Lane', 'Spring Street', 'Garden Avenue', 'Valley Road',
    'Bridge Street', 'Market Street', 'Church Street', 'Mill Road', 'Station Road',
    'High Street', 'King Street', 'Queen Street', 'Princess Avenue', 'Royal Road',
    'Heritage Lane', 'Liberty Avenue',
]
STREETS_DE = [  # German
    'Hauptstraße', 'Bahnhofstraße', 'Schillerstraße', 'Goethestraße', 'Marktstraße',
    'Kirchstraße', 'Schulstraße', 'Bergstraße', 'Talstraße', 'Waldweg',
    'Lindenstraße', 'Gartenstraße', 'Friedrichstraße', 'Kaiserstraße', 'Königstraße',
    'Mühlenweg', 'Sonnenallee', 'Rosenweg', 'Eichenweg', 'Birkenweg',
    'Mozartstraße', 'Beethovenstraße', 'Wagnerstraße', 'Brahmsstraße', 'Heinestraße',
    'Lessingstraße', 'Lindenallee', 'Akazienweg', 'Buchenweg', 'Dorfstraße',
]
STREETS_FR = [  # French
    'Rue de la Paix', 'Avenue Centrale', 'Boulevard du Nord', 'Rue Principale',
    "Rue de l'Église", 'Rue du Marché', 'Avenue des Tilleuls', 'Boulevard Saint-Michel',
    'Rue Nationale', 'Place de la République', "Rue de l'Hôtel-de-Ville",
    'Rue des Lilas', 'Rue des Roses', 'Avenue des Champs', 'Rue Pasteur',
    'Rue Victor Hugo', 'Rue Émile Zola', 'Rue Jean Jaurès', 'Rue Anatole France',
    'Place du Marché', 'Rue de la Gare', 'Rue de la Mairie', 'Rue Saint-Jean',
    'Rue Saint-Pierre', 'Boulevard Pasteur', 'Avenue de la Liberté',
    'Avenue du Général Leclerc', 'Rue des Écoles', 'Rue du Stade', 'Rue Lafayette',
]
STREETS_IT = [  # Italian
    'Via Roma', 'Via Nazionale', 'Via Garibaldi', 'Via Mazzini', 'Via Cavour',
    'Via Dante', 'Via Verdi', 'Via Manzoni', 'Via Marconi', 'Via Galileo',
    'Corso Italia', 'Corso Garibaldi', 'Corso Vittorio Emanuele', 'Piazza del Duomo',
    'Via San Francesco', 'Via San Giovanni', 'Via Santa Maria', 'Via dei Fori',
    'Via Aurelia', 'Via Appia', 'Via della Repubblica', 'Via della Libertà',
    'Via Vittorio Veneto', 'Largo Argentina', 'Via del Corso', 'Via dei Mille',
    'Via XX Settembre', 'Via IV Novembre', 'Via Veneto', 'Via Po',
]
STREETS_ES = [  # Spanish
    'Calle Mayor', 'Calle del Sol', 'Calle Real', 'Calle Nueva', 'Calle de la Iglesia',
    'Avenida de la Constitución', 'Avenida Central', 'Avenida del Mar', 'Paseo Marítimo',
    'Calle de Cervantes', 'Calle de Goya', 'Calle de Velázquez', 'Calle de la Paz',
    'Plaza Mayor', 'Plaza del Pueblo', 'Calle Andalucía', 'Calle Asturias',
    'Calle Galicia', 'Calle Madrid', 'Avenida de España', 'Avenida Libertad',
    'Calle de la Concepción', 'Calle Cádiz', 'Calle Sevilla', 'Calle Aragón',
    'Calle del Olivo', 'Calle del Pino', 'Calle del Río', 'Calle de la Luna',
    'Calle del Sol Naciente',
]
STREETS_PT = [  # Portuguese (PT + BR)
    'Rua da Liberdade', 'Rua do Comércio', 'Avenida Central', 'Avenida da República',
    'Rua das Flores', 'Rua de São João', 'Rua de Santa Maria', 'Rua Nova',
    'Rua Direita', 'Rua Antiga', 'Avenida Atlântica', 'Avenida Brasil',
    'Rua das Palmeiras', 'Rua dos Pinheiros', 'Avenida Paulista', 'Rua Augusta',
    'Rua do Carmo', 'Rua da Paz', 'Avenida Boa Vista', 'Avenida Independência',
    'Rua Sete de Setembro', 'Rua Quinze de Novembro', 'Avenida Beira-Mar',
    'Rua das Acácias', 'Rua dos Coqueiros', 'Rua das Mangueiras', 'Rua das Hortênsias',
    'Avenida Getúlio', 'Avenida Tiradentes', 'Rua Marechal',
]
STREETS_NL = [  # Dutch
    'Hoofdstraat', 'Kerkstraat', 'Schoolstraat', 'Dorpsstraat', 'Stationsstraat',
    'Molenstraat', 'Marktstraat', 'Nieuwstraat', 'Oude Markt', 'Beukenlaan',
    'Eikenlaan', 'Lindenlaan', 'Rozenlaan', 'Tulpenstraat', 'Vondelstraat',
    'Rembrandtstraat', 'Spuistraat', 'Damrak', 'Singel', 'Prinsengracht',
    'Herengracht', 'Keizersgracht', 'Westerstraat', 'Oosterstraat', 'Noorderstraat',
    'Zuiderstraat', 'Wilhelminastraat', 'Beatrixstraat', 'Julianalaan', 'Oranjestraat',
]
STREETS_JP = [  # Japanese romaji
    'Sakura-dori', 'Hinode-dori', 'Asahi-dori', 'Midori-dori', 'Kawakami-dori',
    'Honmachi-dori', 'Chuo-dori', 'Yamanote-dori', 'Aoba-dori', 'Sakuragi-dori',
    'Kasuga-dori', 'Nishi-dori', 'Higashi-dori', 'Minami-dori', 'Kita-dori',
    'Heiwa-dori', 'Showa-dori', 'Hibiya-dori', 'Suehiro-dori', 'Wakaba-dori',
    'Tenjin-dori', 'Yoyogi-dori', 'Akebono-dori', 'Hamamatsu-dori', 'Ginkgo-dori',
    'Momiji-dori', 'Tsubaki-dori', 'Matsu-dori', 'Take-dori', 'Ume-dori',
]
STREETS_CN = [  # Chinese romanized (pinyin)
    'Heping Road', 'Renmin Road', 'Jiefang Road', 'Zhongshan Road', 'Wenhua Road',
    'Xinhua Road', 'Guangming Road', 'Dongfeng Road', 'Beihai Road', 'Nanjing Road',
    'Changjiang Road', 'Huanghe Road', 'Taishan Road', 'Songshan Road', 'Huashan Road',
    'Chunfeng Street', 'Xiandai Boulevard', 'Keji Avenue', 'Jianshe Road', 'Tiyu Road',
    'Yingbin Avenue', 'Xinglong Road', 'Hongqi Street', 'Jianguo Road', 'Fuxing Road',
    'Hepingli Road', 'Qingnian Road', 'Xueyuan Road', 'Daxue Road', 'Yanjiang Road',
]
STREETS_KR = [  # Korean romanized
    'Jongno', 'Daehak-ro', 'Sejong-daero', 'Insadong-gil', 'Cheonggyecheon-ro',
    'Apgujeong-ro', 'Hanbat-daero', 'Hanyang-ro', 'Gangbyeon-daero', 'Hangang-daero',
    'Garosu-gil', 'Itaewon-ro', 'Hongik-ro', 'Dasan-ro', 'Yulgok-ro',
    'Toegye-ro', 'Eulji-ro', 'Namdaemun-ro', 'Jong-ro', 'Yeouido-ro',
    'Olympic-ro', 'Wangsimni-ro', 'Cheongnyangni-ro', 'Mokdong-ro', 'Sinchon-ro',
    'Jamsil-ro', 'Banpo-ro', 'Yeongdong-daero', 'Teheran-ro', 'Munhwa-ro',
]
STREETS_TH = [  # Thai romanized — neutral, not Bangkok-specific
    'Phaholyothin Road', 'Charoen Krung Road', 'Rama IV Road', 'Sathon Tai Road',
    'Witthayu Road', 'Rajadamri Road', 'Si Ayutthaya Road', 'Ratchadamri Road',
    'Lat Phrao Road', 'Ratchaprarop Road', 'Phra Athit Road', 'Khao San Road',
    'Phra Sumen Road', 'Yaowarat Road', 'Pradiphat Road', 'Phaya Thai Road',
    'Henri Dunant Road', 'Ploenchit Road', 'Asok Road', 'Phloen Chit Road',
    'Sukhaphiban Road', 'Bang Na Trat Road', 'Borommarat Chonnani Road',
    'Songkhwae Road', 'Suthep Road', 'Nimmana Haeminda Road', 'Chang Khlan Road',
    'Tha Phae Road', 'Kotchasarn Road', 'Loy Kroh Road',
]
STREETS_VN = [
    'Le Loi Street', 'Nguyen Hue Street', 'Le Duan Street', 'Tran Hung Dao Street',
    'Hai Ba Trung Street', 'Pham Ngu Lao Street', 'Dong Khoi Street', 'Bui Vien Street',
    'Hang Bac Street', 'Hang Bong Street', 'Pho Hue Street', 'Trang Tien Street',
    'Cau Giay Street', 'Kim Ma Street', 'Tay Son Street', 'Giang Vo Street',
    'Hoang Quoc Viet Street', 'Cau Go Street', 'Hang Dao Street', 'Hang Ngang Street',
    'Le Thanh Ton Street', 'Vo Thi Sau Street', 'Hai Ba Trung Boulevard',
    'Pasteur Street', 'Mac Dinh Chi Street', 'Tran Quang Khai Street',
    'Le Lai Street', 'Nam Ky Khoi Nghia Street', 'Nguyen Trai Street',
    'Bach Dang Street',
]
STREETS_HK = [
    'Queens Road', 'Nathan Road', 'Hennessy Road', 'Lockhart Road', 'Hollywood Road',
    'Des Voeux Road', 'Connaught Road', 'Gloucester Road', 'Wong Nai Chung Road',
    'Bonham Road', 'Caine Road', 'Robinson Road', 'MacDonnell Road', 'Cotton Tree Drive',
    'Wyndham Street', 'Lyndhurst Terrace', 'Pottinger Street', 'Wellington Street',
    'Aberdeen Street', 'Stanley Street', 'Des Voeux Road West', 'Belcher Street',
    'Pok Fu Lam Road', 'Mount Davis Road', 'Tin Hau Temple Road', 'Causeway Road',
    'Sing Woo Road', 'Mid-Levels Road', 'Magazine Gap Road', 'Stubbs Road',
]
STREETS_AE = [  # Arabic-style romanized
    'Al Maktoum Road', 'Al Mina Road', 'Al Wahda Street', 'Al Salam Street',
    'Al Khaleej Road', 'Sheikh Zayed Road', 'Al Riyadh Street', 'Al Ittihad Road',
    'Al Falah Street', 'Al Manhal Road', 'Al Jaber Street', 'Al Hisn Road',
    'Al Diyafah Road', 'Al Wasl Road', 'Al Quoz Road', 'Al Garhoud Road',
    'Al Sufouh Road', 'Marina Walk', 'Sea Side Street', 'Beach Street',
    'Corniche Road', 'Heritage Road', 'Cultural Avenue', 'Palace Street',
    'Gold Souk Road', 'Pearl Avenue', 'Date Palm Road', 'Oasis Boulevard',
    'Desert Rose Avenue', 'Falcon Road',
]
STREETS_TR = [
    'Atatürk Caddesi', 'Cumhuriyet Caddesi', 'Bağdat Caddesi', 'İstiklal Caddesi',
    'Barbaros Bulvarı', 'Vatan Caddesi', 'Millet Caddesi', 'Beyoğlu Sokak',
    'Tarlabaşı Bulvarı', 'Karadeniz Caddesi', 'Ege Sokak', 'Marmara Caddesi',
    'Halaskargazi Caddesi', 'Mecidiyeköy Sokak', 'Levent Caddesi', 'Maslak Caddesi',
    'Florya Sokak', 'Yeşilköy Caddesi', 'Bakırköy Sokak', 'Şişli Caddesi',
    'Nişantaşı Sokak', 'Taksim Meydanı', 'Sultanahmet Sokak', 'Sirkeci Caddesi',
    'Eminönü Sokak', 'Galata Sokak', 'Karaköy Caddesi', 'Beşiktaş Sokak',
    'Ortaköy Sokak', 'Üsküdar Caddesi',
]
STREETS_RU = [
    'Tverskaya Street', 'Nevsky Prospekt', 'Arbat Street', 'Petrovka Street',
    'Pushkinskaya Street', 'Gorky Street', 'Sadovaya Street', 'Bolshaya Yakimanka',
    'Leninsky Prospekt', 'Kutuzovsky Prospekt', 'Mira Prospekt', 'Prospekt Vernadskogo',
    'Lubyanka Square', 'Theatre Square', 'Manezhnaya Square', 'Red Square',
    'Pokrovka Street', 'Maroseyka Street', 'Solyanka Street', 'Volkhonka Street',
    'Znamenka Street', 'Vozdvizhenka Street', 'Mokhovaya Street', 'Tverskaya-Yamskaya',
    'Sretenka Street', 'Trubnaya Street', 'Bolshaya Polyanka', 'Yakimanka Street',
    'Ostozhenka Street', 'Prechistenka Street',
]
STREETS_PL = [
    'Krakowskie Przedmieście', 'Marszałkowska', 'Aleje Jerozolimskie', 'Nowy Świat',
    'Świętokrzyska', 'Aleja Niepodległości', 'Aleja Solidarności', 'Złota',
    'Bracka', 'Chmielna', 'Foksal', 'Rondo Dmowskiego', 'Plac Bankowy', 'Senatorska',
    'Miodowa', 'Długa', 'Krzywe Koło', 'Stare Miasto Square', 'Krakowska',
    'Floriańska', 'Grodzka', 'Kanonicza', 'Mikołajska', 'Stradomska',
    'Karmelicka', 'Długi Targ', 'Mariacka', 'Piotrkowska', 'Wajdeloty',
    'Niska',
]
STREETS_GR = [
    'Ermou Street', 'Stadiou Street', 'Panepistimiou Street', 'Akadimias Street',
    'Solonos Street', 'Skoufa Street', 'Patriarchou Ioakeim Street',
    'Vasilissis Sofias Avenue', 'Vasileos Konstantinou Avenue', 'Mesogeion Avenue',
    'Kifissias Avenue', 'Syngrou Avenue', 'Piraeus Avenue', 'Athinas Street',
    'Aiolou Street', 'Adrianou Street', 'Mitropoleos Street', 'Plateia Syntagmatos',
    'Plateia Omonias', 'Apollonos Street', 'Pandrosou Street', 'Lykourgou Street',
    'Stournari Street', 'Themistokleous Street', 'Diligianni Street',
    'Patision Avenue', 'Akadimou Street', 'Voukourestiou Street',
    'Kanari Street', 'Filothei Avenue',
]
STREETS_IL = [
    'Dizengoff Street', 'Allenby Street', 'Rothschild Boulevard', 'Ben Yehuda Street',
    'King George Street', 'Sheinkin Street', 'Bograshov Street', 'Ibn Gabirol Street',
    'Frishman Street', 'Gordon Street', 'Bezalel Yaffe Street', 'Yarkon Street',
    'HaYarkon Street', 'Nachalat Binyamin Street', 'King David Street',
    'Jaffa Road', 'Agrippas Street', 'Strauss Street', 'Sderot HaPalmach',
    'Emek Refaim', 'Hillel Street', 'Shamai Street', 'Ben Hillel Street',
    'Salomon Street', 'Heleni HaMalka', 'Yermiyahu Street', 'David Yellin Street',
    'Bar Ilan Street', 'Pierre Koenig Street', 'HaUmot Boulevard',
]
STREETS_IN = [
    'MG Road', 'Marine Drive', 'Linking Road', 'Brigade Road', 'Park Street',
    'Connaught Place', 'Janpath', 'Rajpath', 'Lodhi Road', 'Mathura Road',
    'Rajiv Chowk Road', 'Hazratganj', 'Mall Road', 'Cunningham Road', 'Residency Road',
    'Mount Road', 'Anna Salai', 'Poonamallee High Road', 'Cathedral Road',
    'Greams Road', 'Sardar Patel Road', 'Connaught Lane', 'Khan Market Road',
    'Bandra-Worli Sea Link Road', 'Andheri-Kurla Road', 'SV Road', 'Hill Road',
    'Carter Road', 'Pali Hill', 'Juhu Tara Road',
]
# Post-Soviet generic (Cyrillic-Latin transliteration, no Moscow-specific landmarks).
# These are common across all former USSR states (Kazakhstan, Kyrgyzstan, Uzbekistan,
# Turkmenistan, Belarus, Ukraine, Georgia, Armenia, Azerbaijan, Tajikistan).
STREETS_POST_SOVIET = [
    'Lenina Street', 'Sovetskaya Street', 'Mira Avenue', 'Pobedy Avenue',
    'Pushkina Street', 'Gagarina Avenue', 'Komsomolskaya Street', 'Tsentralnaya Street',
    'Druzhby Street', 'Nezavisimosti Avenue', 'Universitetskaya Street',
    'Sportivnaya Street', 'Shkolnaya Street', 'Stroiteley Street', 'Molodyozhnaya Street',
    'Yubileynaya Street', 'Oktyabrskaya Street', 'Pervomayskaya Street',
    'Mayakovskogo Street', 'Tolstogo Street', 'Chekhova Street', 'Gorkogo Street',
    'Lermontova Street', 'Kirova Street', 'Karla Marksa Street', 'Frunze Street',
    'Engelsa Street', 'Abay Avenue', 'Navoi Avenue', 'Rudaki Avenue',
]

# Balkan / Eastern-European generic (Slavic-style, no Warsaw-specific landmarks).
# For: Bulgaria, Serbia, Croatia, Bosnia, Montenegro, N. Macedonia, Slovenia,
# Czech Republic, Slovakia.
STREETS_BALKAN = [
    'Glavna ulica', 'Centralna ulica', 'Stara ulica', 'Nova ulica', 'Mladih ulica',
    'Prijateljstva ulica', 'Cara Dusana ulica', 'Kralja Petra ulica', 'Slobode ulica',
    'Mira ulica', 'Skolska ulica', 'Crkvena ulica', 'Trznicka ulica', 'Riblja ulica',
    'Sloboda Avenue', 'Sumadijska ulica', 'Vojvodjanska ulica', 'Dunavska ulica',
    'Brace Krsmanovic ulica', 'Cara Lazara ulica', 'Knez Mihaila ulica',
    'Petrova ulica', 'Branka Radicevica', 'Knjazevska ulica', 'Obilicev Venac',
    'Vasina ulica', 'Skadarska ulica', 'Zmaj Jovina', 'Cara Urosa', 'Strahinjica Bana',
]

# Hungarian — Hungary uses "utca" suffix instead of road/street
STREETS_HU = [
    'Kossuth utca', 'Petofi utca', 'Rakoczi utca', 'Arany Janos utca', 'Jokai utca',
    'Bartok Bela utca', 'Vorosmarty utca', 'Szechenyi utca', 'Ady Endre utca',
    'Dozsa Gyorgy utca', 'Kazinczy utca', 'Tancsics utca', 'Liszt Ferenc utca',
    'Vasut utca', 'Templom utca', 'Iskola utca', 'Korhaz utca', 'Bekek utca',
    'Hosok tere', 'Foter', 'Szabadsag tere', 'Bem Jozsef utca', 'Erkel utca',
    'Munkacsy utca', 'Toldi utca', 'Hunyadi utca', 'Zrinyi utca', 'Pannonia utca',
    'Bocskai utca', 'Pava utca',
]

# Romanian — uses "Strada" prefix
STREETS_RO = [
    'Strada Mare', 'Strada Centrala', 'Strada Republicii', 'Strada Libertatii',
    'Strada Unirii', 'Strada Garii', 'Strada Bisericii', 'Strada Scolii',
    'Strada Mihai Viteazu', 'Strada Stefan cel Mare', 'Strada Eminescu',
    'Strada Caragiale', 'Strada Cosbuc', 'Strada Iorga', 'Strada Brancoveanu',
    'Strada Cuza Voda', 'Strada Decebal', 'Strada Traian', 'Strada Dorobantilor',
    'Strada Avram Iancu', 'Strada Mihail Kogalniceanu', 'Bulevardul Carol',
    'Bulevardul Independentei', 'Strada Pacii', 'Strada Floreasca',
    'Strada Stirbei Voda', 'Strada Bratianu', 'Strada Lipscani',
    'Strada Vitan-Barzesti', 'Strada Magheru',
]

# Bulgarian — uses "ulitsa" / "bulevard" style. Names chosen to be country-generic
# (historical figures common across Bulgaria, common neighborhood-ish names), avoiding
# Sofia-specific landmarks like "Bulevard Dondukov" or "Bulevard Slivnitsa".
STREETS_BG = [
    'Ulitsa Tsar Asen', 'Ulitsa Knyaz Boris', 'Ulitsa Slaveykov', 'Bulevard Levski',
    'Bulevard Botev', 'Ulitsa Rakovski', 'Ulitsa Karavelov', 'Ulitsa Aksakov',
    'Bulevard Patriarh Evtimii', 'Ulitsa Han Krum', 'Bulevard Bulgaria',
    'Ulitsa Tsar Samuil', 'Ulitsa Ivan Vazov', 'Ulitsa Tsar Shishman',
    'Ulitsa General Gurko', 'Ulitsa Hristo Smirnenski', 'Ulitsa Tsar Ivan Asen II',
    'Ulitsa Tsentralna', 'Ulitsa Pirin', 'Ulitsa Rila', 'Ulitsa Vitosha',
    'Ulitsa Cherno More', 'Ulitsa Stara Planina', 'Bulevard Mladost',
    'Bulevard Hristo Botev', 'Bulevard Vasil Levski', 'Bulevard Tsarigradsko shose',
    'Ulitsa Sveti Naum', 'Ulitsa Aprilov', 'Ulitsa Khan Asparuh',
]

# Latvian — "iela" (street), "bulvaris" (boulevard), "laukums" (square)
STREETS_LV = [
    'Liela iela', 'Brivibas iela', 'Elizabetes iela', 'Kalku iela',
    'Aspazijas bulvaris', 'Maza iela', 'Skolas iela', 'Berzu iela',
    'Egles iela', 'Ozolu iela', 'Liepu iela', 'Klusa iela',
    'Rigas iela', 'Brivibas bulvaris', 'Valdemara iela', 'Lacplesa iela',
    'Marijas iela', 'Tallinas iela', 'Krisjana Barona iela', 'Akademijas laukums',
    'Esplanade laukums', 'Stacijas laukums', 'Saules iela', 'Pasta iela',
    'Vecpilsetas iela', 'Tirgus iela', 'Krasta iela', 'Strelnieku iela',
    'Dzelzavas iela', 'Janvara iela',
]

# Lithuanian — "gatve" (street), "prospektas" (avenue), "aikste" (square)
STREETS_LT = [
    'Gedimino prospektas', 'Vilniaus gatve', 'Kauno gatve', 'Lukiskiu aikste',
    'Pilies gatve', 'Pylimo gatve', 'Antakalnio gatve', 'Sodu gatve',
    'Konstitucijos prospektas', 'Vasaros gatve', 'Sauletekio aleja',
    'Laisves aleja', 'Naugarduko gatve', 'Savanoriu prospektas', 'Subaciaus gatve',
    'Aukstaiciu gatve', 'Birutes gatve', 'Trakai gatve', 'Mindaugo gatve',
    'Vaidoto gatve', 'Liepu aleja', 'Stulginskio gatve', 'Donelaicio gatve',
    'Kovo 11-osios gatve', 'Maironio gatve', 'Sapiegos gatve',
    'Veiveriu gatve', 'Tilto gatve', 'Universiteto gatve', 'Pranciskonu gatve',
]

# Estonian — "tanav" (street), "maantee" (road), "puiestee" (avenue)
STREETS_EE = [
    'Vana tanav', 'Parnu maantee', 'Estonia puiestee', 'Toompuiestee',
    'Tartu maantee', 'Kalevipoja allee', 'Voorimehe tanav', 'Roosi tanav',
    'Sauna tanav', 'Kreutzwaldi tanav', 'Liivalaia tanav', 'Pikk tanav',
    'Vene tanav', 'Mere puiestee', 'Suur-Karja tanav', 'Kuninga tanav',
    'Vabaduse valjak', 'Raekoja plats', 'Karu tanav', 'Maakri tanav',
    'Narva maantee', 'Sopruse puiestee', 'Roosikrantsi tanav', 'Munga tanav',
    'Lossi plats', 'Kunderi tanav', 'Tartu tanav', 'Kentmanni tanav',
    'Riia tanav', 'Lai tanav',
]

# Czech / Slovak — "ulice" (street), "namesti" (square), "trida" (avenue)
STREETS_CZ_SK = [
    'Hlavni trida', 'Narodni trida', 'Vaclavske namesti', 'Stara ulice',
    'Nova ulice', 'Kostelni ulice', 'Skolni ulice', 'Nadrazni ulice',
    'Hradebni ulice', 'Mostni ulice', 'Parkova ulice', 'Lesna ulice',
    'Hornicka ulice', 'Mlynska ulice', 'Husova ulice', 'Komenskeho ulice',
    'Palackeho namesti', 'Masarykovo namesti', 'Tylovo namesti', 'Smetanova ulice',
    'Dvorakova ulice', 'Janackova ulice', 'Krizikova ulice', 'Resslova ulice',
    'Stefanikova ulice', 'Benesova ulice', 'Hviezdoslavovo namestie',
    'SNP namestie', 'Stara cesta', 'Jiraskovo namesti',
]

# Yugoslav successor states (Serbia/Croatia/Bosnia/Montenegro/N. Macedonia/Slovenia)
# Uses "ulica" Slavic convention without Czech/Polish-specific landmarks
STREETS_YUGOSLAV = [
    'Glavna ulica', 'Centralna ulica', 'Stara ulica', 'Nova ulica',
    'Mladih ulica', 'Prijateljstva ulica', 'Slobode ulica', 'Mira ulica',
    'Skolska ulica', 'Crkvena ulica', 'Trznicka ulica', 'Riblja ulica',
    'Sloboda Avenue', 'Sumadijska ulica', 'Vojvodjanska ulica',
    'Petrova ulica', 'Branka Radicevica', 'Knjazevska ulica', 'Obilicev Venac',
    'Vasina ulica', 'Skadarska ulica', 'Zmaj Jovina', 'Strahinjica Bana',
    'Bulevar mira', 'Bulevar mladih', 'Trg slobode', 'Trg republike',
    'Kralja Petra ulica', 'Cara Dusana ulica', 'Cara Lazara ulica',
    'Karadjordjeva ulica',
]

# Cypriot — mostly English/Greek mixed, avoiding Athens-specific landmarks
STREETS_CY = [
    'Makarios Avenue', 'Stasinou Avenue', 'Themistocles Dervis Street',
    'Spyrou Kyprianou Avenue', 'Larnaca Avenue', 'Limassol Avenue',
    'Eleftheria Square', 'Faneromenis Street', 'Onasagorou Street',
    'Lidras Street', 'Athinodorou Street', 'Anastasi Street',
    'Saint Lazarus Square', 'Tombs of the Kings Avenue', 'Posidonos Avenue',
    'Athens Street', 'Heroes Street', 'Independence Avenue',
    'Archbishop Makarios III Avenue', 'Acropolis Avenue', 'Olympion Street',
    'Solomon Street', 'Pindarou Street', 'Vyzantiou Street', 'Aristotelous Street',
    'Plato Street', 'Sokrates Street', 'Homer Street', 'Pythagoras Street',
    'Aristides Street',
]

# Country → (template, street_pool)
LOCALE = {
    # ---- English-speaking ----
    'United States':  ('{num} {street}, {city}', STREETS_EN),
    'Canada':         ('{num} {street}, {city}', STREETS_EN),
    'United Kingdom': ('{num} {street}, {city}', STREETS_EN),
    'Australia':      ('{num} {street}, {city}', STREETS_EN),
    'New Zealand':    ('{num} {street}, {city}', STREETS_EN),
    'Ireland':        ('{num} {street}, {city}', STREETS_EN),
    'South Africa':   ('{num} {street}, {city}', STREETS_EN),
    'Singapore':      ('{num} {street}, {city}', STREETS_EN),
    'Philippines':    ('{num} {street}, {city}', STREETS_EN),
    'Kenya':          ('{num} {street}, {city}', STREETS_EN),
    'Nigeria':        ('{num} {street}, {city}', STREETS_EN),
    'Ghana':          ('{num} {street}, {city}', STREETS_EN),
    'Ethiopia':       ('{num} {street}, {city}', STREETS_EN),
    'Pakistan':       ('{num} {street}, {city}', STREETS_EN),
    'Bangladesh':     ('{num} {street}, {city}', STREETS_EN),
    'Sri Lanka':      ('{num} {street}, {city}', STREETS_EN),

    # ---- European ----
    'Germany':        ('{street} {num}, {city}', STREETS_DE),
    'Austria':        ('{street} {num}, {city}', STREETS_DE),
    'Switzerland':    ('{street} {num}, {city}', STREETS_DE),
    'France':         ('{num} {street}, {city}',  STREETS_FR),
    'Italy':          ('{street} {num}, {city}', STREETS_IT),
    'Spain':          ('{street}, {num}, {city}', STREETS_ES),
    'Portugal':       ('{street}, {num}, {city}', STREETS_PT),
    'Brazil':         ('{street}, {num}, {city}', STREETS_PT),
    'Netherlands':    ('{street} {num}, {city}', STREETS_NL),
    'Belgium':        ('{street} {num}, {city}', STREETS_NL),
    'Sweden':         ('{street} {num}, {city}', STREETS_DE),
    'Denmark':        ('{street} {num}, {city}', STREETS_DE),
    'Norway':         ('{street} {num}, {city}', STREETS_DE),
    'Finland':        ('{street} {num}, {city}', STREETS_DE),
    'Poland':         ('{street} {num}, {city}', STREETS_PL),
    'Czech Republic': ('{street} {num}, {city}', STREETS_CZ_SK),
    'Czechia':        ('{street} {num}, {city}', STREETS_CZ_SK),
    'Greece':         ('{num} {street}, {city}', STREETS_GR),
    'Hungary':        ('{street} {num}, {city}', STREETS_HU),
    'Romania':        ('{street} {num}, {city}', STREETS_RO),

    # ---- East / South Asia ----
    'Japan':       ('{num} {street}, {city}', STREETS_JP),
    'China':       ('{num} {street}, {city}', STREETS_CN),
    'Taiwan':      ('{num} {street}, {city}', STREETS_CN),
    'Hong Kong':   ('{num} {street}, {city}', STREETS_HK),
    'Macau':       ('{num} {street}, {city}', STREETS_CN),
    'South Korea': ('{num} {street}, {city}', STREETS_KR),
    'Korea':       ('{num} {street}, {city}', STREETS_KR),
    'Thailand':    ('{num} {street}, {city}', STREETS_TH),
    'Vietnam':     ('{num} {street}, {city}', STREETS_VN),
    'Indonesia':   ('{num} {street}, {city}', STREETS_EN),
    'Malaysia':    ('{num} {street}, {city}', STREETS_EN),
    'India':       ('{num} {street}, {city}', STREETS_IN),
    'Nepal':       ('{num} {street}, {city}', STREETS_EN),

    # ---- Middle East / North Africa ----
    'UAE':                  ('{num} {street}, {city}', STREETS_AE),
    'United Arab Emirates': ('{num} {street}, {city}', STREETS_AE),
    'Saudi Arabia':         ('{num} {street}, {city}', STREETS_AE),
    'Qatar':                ('{num} {street}, {city}', STREETS_AE),
    'Israel':               ('{street} {num}, {city}', STREETS_IL),
    'Turkey':               ('{street} {num}, {city}', STREETS_TR),
    'Egypt':                ('{num} {street}, {city}', STREETS_AE),
    'Jordan':               ('{num} {street}, {city}', STREETS_AE),
    'Lebanon':              ('{num} {street}, {city}', STREETS_AE),
    'Morocco':              ('{num} {street}, {city}', STREETS_FR),
    'Tunisia':              ('{num} {street}, {city}', STREETS_FR),
    'Algeria':              ('{num} {street}, {city}', STREETS_FR),

    # ---- Latin America (Spanish) ----
    'Mexico':    ('{street}, {num}, {city}', STREETS_ES),
    'Argentina': ('{street} {num}, {city}', STREETS_ES),
    'Chile':     ('{street} {num}, {city}', STREETS_ES),
    'Colombia':  ('{street}, {num}, {city}', STREETS_ES),
    'Peru':      ('{street}, {num}, {city}', STREETS_ES),
    'Ecuador':   ('{street}, {num}, {city}', STREETS_ES),
    'Uruguay':   ('{street} {num}, {city}', STREETS_ES),
    'Cuba':      ('{street}, {num}, {city}', STREETS_ES),
    'Costa Rica':('{street}, {num}, {city}', STREETS_ES),
    'Panama':    ('{street}, {num}, {city}', STREETS_ES),

    # ---- Russia / former USSR ----
    'Russia':       ('{street} {num}, {city}', STREETS_RU),
    'Ukraine':      ('{street} {num}, {city}', STREETS_POST_SOVIET),
    'Kazakhstan':   ('{street} {num}, {city}', STREETS_POST_SOVIET),
    'Belarus':      ('{street} {num}, {city}', STREETS_POST_SOVIET),
    'Uzbekistan':   ('{street} {num}, {city}', STREETS_POST_SOVIET),
    'Georgia':      ('{street} {num}, {city}', STREETS_POST_SOVIET),
    'Azerbaijan':   ('{street} {num}, {city}', STREETS_POST_SOVIET),
    'Armenia':      ('{street} {num}, {city}', STREETS_POST_SOVIET),
    'Kyrgyzstan':   ('{street} {num}, {city}', STREETS_POST_SOVIET),
    'Turkmenistan': ('{street} {num}, {city}', STREETS_POST_SOVIET),

    # ---- Slavic / Baltic / Balkan ----
    'Bulgaria':       ('{street} {num}, {city}', STREETS_BG),
    'Slovakia':       ('{street} {num}, {city}', STREETS_CZ_SK),
    'Croatia':        ('{street} {num}, {city}', STREETS_YUGOSLAV),
    'Serbia':         ('{street} {num}, {city}', STREETS_YUGOSLAV),
    'Montenegro':     ('{street} {num}, {city}', STREETS_YUGOSLAV),
    'Bosnia and Herzegovina': ('{street} {num}, {city}', STREETS_YUGOSLAV),
    'North Macedonia':('{street} {num}, {city}', STREETS_YUGOSLAV),
    'Slovenia':       ('{street} {num}, {city}', STREETS_YUGOSLAV),
    'Latvia':         ('{street} {num}, {city}', STREETS_LV),
    'Lithuania':      ('{street} {num}, {city}', STREETS_LT),
    'Estonia':        ('{street} {num}, {city}', STREETS_EE),

    # ---- Latin America (additional) ----
    'Venezuela':          ('{street}, {num}, {city}', STREETS_ES),
    'Dominican Republic': ('{street}, {num}, {city}', STREETS_ES),
    'Guatemala':          ('{street}, {num}, {city}', STREETS_ES),
    'El Salvador':        ('{street}, {num}, {city}', STREETS_ES),
    'Paraguay':           ('{street}, {num}, {city}', STREETS_ES),
    'Bolivia':            ('{street}, {num}, {city}', STREETS_ES),
    'Belize':             ('{num} {street}, {city}', STREETS_EN),
    'Aruba':              ('{street} {num}, {city}', STREETS_NL),
    'Curaçao':            ('{street} {num}, {city}', STREETS_NL),
    'Jamaica':            ('{num} {street}, {city}', STREETS_EN),
    'Saint Lucia':        ('{num} {street}, {city}', STREETS_EN),
    'Cayman Islands':     ('{num} {street}, {city}', STREETS_EN),
    'Turks and Caicos Islands': ('{num} {street}, {city}', STREETS_EN),
    'Puerto Rico':        ('{street}, {num}, {city}', STREETS_ES),
    'Martinique':         ('{num} {street}, {city}', STREETS_FR),
    'Guadeloupe':         ('{num} {street}, {city}', STREETS_FR),
    'French Guiana':      ('{num} {street}, {city}', STREETS_FR),
    'French Polynesia':   ('{num} {street}, {city}', STREETS_FR),
    'Réunion':            ('{num} {street}, {city}', STREETS_FR),

    # ---- Africa (additional) ----
    'Tanzania':     ('{num} {street}, {city}', STREETS_EN),
    'Uganda':       ('{num} {street}, {city}', STREETS_EN),
    'Zambia':       ('{num} {street}, {city}', STREETS_EN),
    'Zimbabwe':     ('{num} {street}, {city}', STREETS_EN),
    'Botswana':     ('{num} {street}, {city}', STREETS_EN),
    'Rwanda':       ('{num} {street}, {city}', STREETS_EN),
    'Mauritius':    ('{num} {street}, {city}', STREETS_EN),
    'Seychelles':   ('{num} {street}, {city}', STREETS_EN),
    'Mozambique':   ('{street}, {num}, {city}', STREETS_PT),
    'Angola':       ('{street}, {num}, {city}', STREETS_PT),
    'Cape Verde':   ('{street}, {num}, {city}', STREETS_PT),
    'São Tomé and Principe': ('{street}, {num}, {city}', STREETS_PT),
    'Niger':        ('{num} {street}, {city}', STREETS_FR),
    'Chad':         ('{num} {street}, {city}', STREETS_FR),
    'Mali':         ('{num} {street}, {city}', STREETS_FR),
    'Mauritania':   ('{num} {street}, {city}', STREETS_FR),
    'Djibouti':     ('{num} {street}, {city}', STREETS_FR),
    'South Sudan':  ('{num} {street}, {city}', STREETS_EN),
    'Democratic Republic of the Congo': ('{num} {street}, {city}', STREETS_FR),

    # ---- Middle East / additional ----
    'Iraq':       ('{num} {street}, {city}', STREETS_AE),
    'Kuwait':     ('{num} {street}, {city}', STREETS_AE),
    'Oman':       ('{num} {street}, {city}', STREETS_AE),
    'Cyprus':     ('{num} {street}, {city}', STREETS_CY),

    # ---- South / Southeast Asia (additional) ----
    'Myanmar':    ('{num} {street}, {city}', STREETS_EN),
    'Cambodia':   ('{num} {street}, {city}', STREETS_EN),
    'Brunei':     ('{num} {street}, {city}', STREETS_EN),

    # ---- Pacific / others ----
    'Solomon Islands': ('{num} {street}, {city}', STREETS_EN),
    'Guam':            ('{num} {street}, {city}', STREETS_EN),
    'Iceland':         ('{street} {num}, {city}', STREETS_DE),
    'Malta':           ('{num} {street}, {city}', STREETS_EN),
    'Luxembourg':      ('{num} {street}, {city}', STREETS_FR),
    'Gibraltar':       ('{num} {street}, {city}', STREETS_EN),
    'Costa Rica':      ('{street}, {num}, {city}', STREETS_ES),
}

# Fallback for any country not in LOCALE
FALLBACK_TEMPLATE = '{num} {street}, {city}'
FALLBACK_STREETS = STREETS_EN

# ----------------------------------------------------------------------------
# Field-level fix functions
# ----------------------------------------------------------------------------
def fix_rating_denominator(about: str) -> str:
    """Bug H1: 'rating of 7.6/5' -> 'rating of 7.6/10'."""
    return re.sub(r'(rating of \d+(?:\.\d+)?)/5', r'\1/10', about)


def fix_star_truncation(about: str, numeric_star: float) -> str:
    """Bug H2: 'is a 2-star hotel' -> 'is a 2.5-star hotel' when numeric_star=2.5."""
    if numeric_star == int(numeric_star):
        replacement = f'is a {int(numeric_star)}-star hotel'
    else:
        replacement = f'is a {numeric_star}-star hotel'
    return re.sub(r'is a \d+(?:\.\d+)?-star hotel', replacement, about)


def regen_address(country: str, city: str, rng: np.random.Generator) -> str:
    """Bug H4 / A1: regenerate address using locale-aware template + neutral street."""
    tmpl, streets = LOCALE.get(country, (FALLBACK_TEMPLATE, FALLBACK_STREETS))
    return tmpl.format(
        num=int(rng.integers(1, 9999)),
        street=streets[rng.integers(0, len(streets))],
        city=city,
    )


def rewrite_overview_price(overview: str, new_price: float) -> str:
    """Bug A5: an attraction overview states its ticket price in two different phrasings --
    'Entry fee is X USD' near the start and ', entry X USD' at the end. Both must track the
    numeric ticket_price column or the record contradicts itself."""
    text = re.sub(r'(Entry fee is )[\d.]+( USD)', rf'\g<1>{new_price}\g<2>', overview)
    text = re.sub(r'(, entry )[\d.]+( USD)', rf'\g<1>{new_price}\g<2>', text)
    return text


def fix_attraction_open_hours(overview: str, numeric_open: str) -> str:
    """Bug A3 was a false alarm: the original audit regex `Open (hh:mm)-(hh:mm)` only captured the
    first segment of multi-segment open hours (e.g. lunch+dinner like "11:30-14:30, 17:30-20:30"),
    flagging these as mismatches when in fact the overview already contained the full string.
    No fix is needed — function returns overview unchanged."""
    return overview


# ----------------------------------------------------------------------------
# Per-file processors
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Airport timezone table (Bug F1)
# ----------------------------------------------------------------------------
# flights.csv carries no date, no duration and no timezone: only two local HH:MM stamps. A
# flight's real duration is therefore arrival_local - departure_local - (tz_arr - tz_dep), and
# judging it needs each airport's true UTC offset. The original F1 approximated this from the
# country code, which is wrong in both directions -- it split single-timezone Germany across a
# meridian and excluded China as "multi-timezone" when the CAAC publishes every domestic
# schedule in Beijing time.
#
# Offsets below are minutes east of UTC, resolved per airport from its coordinates via IANA
# tzdata (generated once with timezonefinder 6.5.9; the table is inlined so the build stays
# offline and auditable). Each entry is (standard, daylight) sampled at 2025-01-15 and
# 2025-07-15. Every Chinese airport is pinned to +480 including Urumqi, whose IANA zone is
# Asia/Urumqi (+06) but whose flights are scheduled in Beijing time.
#
# The source snapshot is agoda_direct_20250801, so the daylight column is the offset in force
# on the snapshot date -- verified identical at 2025-07-15 and 2025-08-01 for all 392 airports.
AIRPORT_TZ = {
    'ABQ': (-420, -360), 'ACC': (0, 0), 'ACE': (0, 60), 'ADB': (180, 180),
    'ADD': (180, 180), 'ADL': (630, 570), 'AEP': (-180, -180), 'AGA': (60, 60),
    'AGP': (60, 120), 'AHB': (180, 180), 'AKL': (780, 720), 'ALA': (300, 300),
    'ALC': (60, 120), 'ALG': (60, 60), 'AMM': (180, 180), 'AMS': (60, 120),
    'ANC': (-540, -480), 'ARN': (60, 120), 'ASB': (300, 300), 'ASU': (-180, -180),
    'ATH': (120, 180), 'ATL': (-300, -240), 'AUA': (-240, -240), 'AUH': (240, 240),
    'AUS': (-360, -300), 'AYT': (180, 180), 'BDL': (-300, -240), 'BDS': (60, 120),
    'BEG': (60, 120), 'BEL': (-180, -180), 'BEY': (120, 180), 'BFS': (0, 60),
    'BGO': (60, 120), 'BGW': (180, 180), 'BGY': (60, 120), 'BHX': (0, 60),
    'BJV': (180, 180), 'BKK': (420, 420), 'BKO': (0, 0), 'BLL': (60, 120),
    'BLQ': (60, 120), 'BLR': (330, 330), 'BNA': (-360, -300), 'BNE': (600, 600),
    'BOD': (60, 120), 'BOG': (-300, -300), 'BOJ': (120, 180), 'BOM': (330, 330),
    'BOS': (-300, -240), 'BPN': (480, 480), 'BPS': (-180, -180), 'BRI': (60, 120),
    'BRU': (60, 120), 'BSB': (-180, -180), 'BSL': (60, 120), 'BTS': (60, 120),
    'BUD': (60, 120), 'BUF': (-300, -240), 'BWI': (-300, -240), 'BWN': (480, 480),
    'BZE': (-360, -360), 'CAG': (60, 120), 'CAI': (120, 180), 'CAN': (480, 480),
    'CAY': (-180, -180), 'CCS': (-240, -240), 'CCU': (330, 330), 'CDG': (60, 120),
    'CEB': (480, 480), 'CGK': (420, 420), 'CGN': (60, 120), 'CGO': (480, 480),
    'CGQ': (480, 480), 'CHC': (780, 720), 'CJU': (540, 540), 'CKG': (480, 480),
    'CLE': (-300, -240), 'CLT': (-300, -240), 'CMB': (330, 330), 'CMH': (-300, -240),
    'CMN': (60, 60), 'CNF': (-180, -180), 'CNX': (420, 420), 'COK': (330, 330),
    'CSX': (480, 480), 'CTA': (60, 120), 'CTS': (540, 540), 'CTU': (480, 480),
    'CUN': (-300, -300), 'CUR': (-240, -240), 'CUZ': (-300, -300), 'CVG': (-300, -240),
    'CZM': (-300, -300), 'DAC': (360, 360), 'DAR': (180, 180), 'DCA': (-300, -240),
    'DEL': (330, 330), 'DEN': (-420, -360), 'DFW': (-360, -300), 'DGO': (-360, -360),
    'DLC': (480, 480), 'DLM': (180, 180), 'DMK': (420, 420), 'DOH': (180, 180),
    'DPS': (480, 480), 'DTW': (-300, -240), 'DUB': (0, 60), 'DUS': (60, 120),
    'DVO': (480, 480), 'DWC': (240, 240), 'DXB': (240, 240), 'EBB': (180, 180),
    'EDI': (0, 60), 'EIN': (60, 120), 'ESB': (180, 180), 'EVN': (240, 240),
    'EWR': (-300, -240), 'EZE': (-180, -180), 'FAO': (0, 60), 'FCO': (60, 120),
    'FDF': (-240, -240), 'FEZ': (60, 60), 'FIH': (60, 60), 'FLL': (-300, -240),
    'FLN': (-180, -180), 'FLR': (60, 120), 'FNC': (0, 60), 'FOC': (480, 480),
    'FOR': (-180, -180), 'FRA': (60, 120), 'FRU': (360, 360), 'FUK': (540, 540),
    'GBE': (120, 120), 'GCM': (-300, -300), 'GDL': (-360, -360), 'GDN': (60, 120),
    'GIB': (60, 120), 'GIG': (-180, -180), 'GMP': (540, 540), 'GOA': (60, 120),
    'GOT': (60, 120), 'GRU': (-180, -180), 'GUA': (-360, -360), 'GUM': (600, 600),
    'GVA': (60, 120), 'GYD': (240, 240), 'GYE': (-300, -300), 'HAJ': (60, 120),
    'HAK': (480, 480), 'HAM': (60, 120), 'HAN': (420, 420), 'HEL': (120, 180),
    'HET': (480, 480), 'HGH': (480, 480), 'HIR': (660, 660), 'HKG': (480, 480),
    'HKT': (420, 420), 'HMB': (120, 180), 'HND': (540, 540), 'HNL': (-600, -600),
    'HRB': (480, 480), 'HRE': (120, 120), 'HYD': (330, 330), 'IAH': (-360, -300),
    'IBZ': (60, 120), 'ICN': (540, 540), 'IND': (-300, -240), 'ISB': (300, 300),
    'IST': (180, 180), 'ITM': (540, 540), 'JAX': (-300, -240), 'JFK': (-300, -240),
    'JIB': (180, 180), 'JUB': (120, 120), 'KEF': (0, 0), 'KGL': (120, 120),
    'KHH': (480, 480), 'KHI': (300, 300), 'KHN': (480, 480), 'KIN': (-300, -300),
    'KIX': (540, 540), 'KMG': (480, 480), 'KOJ': (540, 540), 'KRK': (60, 120),
    'KTM': (345, 345), 'KUL': (480, 480), 'KWE': (480, 480), 'KWI': (180, 180),
    'KWL': (480, 480), 'LAD': (60, 60), 'LAS': (-480, -420), 'LAX': (-480, -420),
    'LCA': (120, 180), 'LEJ': (60, 120), 'LGA': (-300, -240), 'LGW': (0, 60),
    'LHR': (0, 60), 'LHW': (480, 480), 'LIM': (-300, -300), 'LIR': (-360, -360),
    'LIS': (0, 60), 'LOS': (60, 60), 'LPA': (0, 60), 'LRM': (-240, -240),
    'LTN': (0, 60), 'LUN': (120, 120), 'LUX': (60, 120), 'LYS': (60, 120),
    'MAA': (330, 330), 'MAD': (60, 120), 'MAN': (0, 60), 'MAO': (-240, -240),
    'MBA': (180, 180), 'MCI': (-360, -300), 'MCO': (-300, -240), 'MCT': (240, 240),
    'MDL': (390, 390), 'MDW': (-360, -300), 'MED': (180, 180), 'MEL': (660, 600),
    'MEM': (-360, -300), 'MEX': (-360, -360), 'MFM': (480, 480), 'MIA': (-300, -240),
    'MID': (-360, -360), 'MKE': (-360, -300), 'MLA': (60, 120), 'MNL': (480, 480),
    'MPM': (120, 120), 'MRS': (60, 120), 'MRU': (240, 240), 'MSP': (-360, -300),
    'MSY': (-360, -300), 'MTY': (-360, -360), 'MUC': (60, 120), 'MVD': (-180, -180),
    'MXP': (60, 120), 'MZT': (-420, -420), 'NAP': (60, 120), 'NBO': (180, 180),
    'NCE': (60, 120), 'NDJ': (60, 60), 'NGB': (480, 480), 'NGO': (540, 540),
    'NIM': (60, 60), 'NKC': (0, 0), 'NKG': (480, 480), 'NKM': (540, 540),
    'NNG': (480, 480), 'NRT': (540, 540), 'NUE': (60, 120), 'OAK': (-480, -420),
    'OGG': (-600, -600), 'OKA': (540, 540), 'OKC': (-360, -300), 'OKD': (540, 540),
    'OMA': (-360, -300), 'ONT': (-480, -420), 'OPO': (0, 60), 'ORD': (-360, -300),
    'ORF': (-300, -240), 'ORY': (60, 120), 'OSL': (60, 120), 'OTP': (120, 180),
    'PBI': (-300, -240), 'PDL': (-60, 0), 'PDX': (-480, -420), 'PEK': (480, 480),
    'PER': (480, 480), 'PHL': (-300, -240), 'PHX': (-420, -420), 'PIT': (-300, -240),
    'PKX': (480, 480), 'PLS': (-300, -240), 'PMI': (60, 120), 'PMO': (60, 120),
    'PNH': (420, 420), 'PPT': (-600, -600), 'PRG': (60, 120), 'PSA': (60, 120),
    'PTP': (-240, -240), 'PTY': (-300, -300), 'PUS': (540, 540), 'PVD': (-300, -240),
    'PVG': (480, 480), 'PVR': (-360, -360), 'PWM': (-300, -240), 'RAI': (-60, -60),
    'RAK': (60, 60), 'RBA': (60, 60), 'RDU': (-300, -240), 'RGN': (390, 390),
    'RIC': (-300, -240), 'RIX': (120, 180), 'RNO': (-480, -420), 'RSW': (-300, -240),
    'RUH': (180, 180), 'RUN': (240, 240), 'SAL': (-360, -360), 'SAN': (-480, -420),
    'SAT': (-360, -300), 'SAV': (-300, -240), 'SAW': (180, 180), 'SCL': (-180, -240),
    'SCQ': (60, 120), 'SDF': (-300, -240), 'SDJ': (540, 540), 'SDQ': (-240, -240),
    'SEA': (-480, -420), 'SEZ': (240, 240), 'SFB': (-300, -240), 'SFO': (-480, -420),
    'SGN': (420, 420), 'SHA': (480, 480), 'SHE': (480, 480), 'SHJ': (240, 240),
    'SIN': (480, 480), 'SJC': (-480, -420), 'SJJ': (60, 120), 'SJO': (-360, -360),
    'SJU': (-240, -240), 'SKG': (120, 180), 'SKP': (60, 120), 'SLC': (-420, -360),
    'SMF': (-480, -420), 'SNA': (-480, -420), 'SNN': (0, 60), 'SOF': (120, 180),
    'SRQ': (-300, -240), 'SSA': (-180, -180), 'STL': (-360, -300), 'STN': (0, 60),
    'STR': (60, 120), 'SUB': (420, 420), 'SVG': (60, 120), 'SYD': (660, 600),
    'SYR': (-300, -240), 'SYX': (480, 480), 'SZX': (480, 480), 'TAO': (480, 480),
    'TAS': (300, 300), 'TBS': (240, 240), 'TFS': (0, 60), 'TFU': (480, 480),
    'TGD': (60, 120), 'TLL': (120, 180), 'TLS': (60, 120), 'TLV': (120, 180),
    'TMS': (0, 0), 'TNA': (480, 480), 'TNG': (60, 60), 'TOS': (60, 120),
    'TPA': (-300, -240), 'TPE': (480, 480), 'TRD': (60, 120), 'TRN': (60, 120),
    'TRV': (330, 330), 'TSN': (480, 480), 'TUL': (-360, -300), 'TUN': (60, 60),
    'TYN': (480, 480), 'UIO': (-300, -300), 'UPG': (480, 480), 'URC': (480, 480),
    'UVF': (-240, -240), 'VAR': (120, 180), 'VCE': (60, 120), 'VIE': (60, 120),
    'VIX': (-180, -180), 'VNO': (120, 180), 'VRN': (60, 120), 'VVI': (-240, -240),
    'WAW': (60, 120), 'WLG': (780, 720), 'WNZ': (480, 480), 'WUH': (480, 480),
    'XIY': (480, 480), 'XMN': (480, 480), 'YEG': (-420, -360), 'YHZ': (-240, -180),
    'YNT': (480, 480), 'YOW': (-300, -240), 'YQB': (-300, -240), 'YUL': (-300, -240),
    'YVR': (-480, -420), 'YWG': (-360, -300), 'YYC': (-420, -360), 'YYT': (-210, -150),
    'YYZ': (-300, -240), 'ZAG': (60, 120), 'ZNZ': (180, 180), 'ZRH': (60, 120),
}
DST_STD, DST_SUM = 0, 1   # index into an AIRPORT_TZ entry


# ----------------------------------------------------------------------------
# Bug C1 — coordinate audit and repair
# ----------------------------------------------------------------------------
# Hotel and attraction coordinates were sampled from a per-city bounding box. For a minority
# of cities that box was mis-parameterised: either anchored on the wrong place (Ankara's
# records sit within 300 m of each other, 714 km from Ankara) or stretched to a
# prefecture-scale extent (Riyadh's median record lies 239 km from the centre). Both corrupt
# D6, which derives inter-entity travel time from these coordinates.
#
# Audit  — each city is checked against its OurAirports reference point, matched on
# (city, country) so that same-name cities in different countries (San Jose CR vs San Jose US)
# cannot cross-contaminate; a city with several airports takes the nearest.
# Repair — a failing city's cloud is re-anchored onto its reference point and its dispersion
# rescaled to a target drawn from the empirical dispersion of the cities that passed, which
# keeps the repaired cities statistically indistinguishable from the untouched ones. Relative
# structure within a city is preserved, and hotels and attractions share one transform so the
# two domains stay in a common frame.
#
# Coordinates take no part in D0-src (it matches on name + city), so this leaves entity
# grounding and every existing query annotation untouched. Only D6 moves.
DISPLACED_KM = 80.0   # cloud centre this far from the city's airport => wrong anchor
DISPERSED_KM = 60.0   # 90th-percentile radius this large => box stretched beyond a metro
DEG_KM = 111.32       # km per degree of latitude


def _norm_key(s) -> str:
    """Fold accents and case so 'Asunción' matches 'Asuncion' across the two tables."""
    s = unicodedata.normalize('NFKD', str(s))
    return ''.join(c for c in s if not unicodedata.combining(c)).lower().strip()


def _load_airport_refs() -> dict:
    """(city, country) -> [(lat, lon), ...] from the OurAirports-derived airport table."""
    df = pd.read_csv(os.path.join(IN_FLIGHT, 'VALID_airports_id.csv'))
    refs = defaultdict(list)
    for r in df.itertuples():
        refs[(_norm_key(r.city), _norm_key(r.country))].append(
            (float(r.latitude_deg), float(r.longitude_deg)))
    return dict(refs)


def _city_files(city: str) -> list:
    """The (path, lat_col, lon_col) triples holding one city's coordinates."""
    out = []
    for d, suffix in ((IN_HOTEL, '_hotel.csv'), (IN_ATTR, '_attraction.csv')):
        p = os.path.join(d, city + suffix)
        if os.path.exists(p):
            out.append(p)
    return out


def _read_city_cloud(city: str):
    """Pooled (lat, lon) of every hotel and attraction in a city, plus its canonical name and
    country. The canonical name comes from the `city_name` column, not the filename: filenames
    escape an apostrophe to an underscore (`Xi_an.csv` holds `Xi'an`), which no longer matches
    the airport table."""
    lats, lons, country, name = [], [], None, None
    for p in _city_files(city):
        d = pd.read_csv(p)
        if 'latitude' not in d.columns or 'longitude' not in d.columns:
            continue
        lats.append(d['latitude'].to_numpy(dtype=float))
        lons.append(d['longitude'].to_numpy(dtype=float))
        if len(d):
            if country is None and 'country' in d.columns:
                country = str(d['country'].iloc[0])
            if name is None and 'city_name' in d.columns:
                name = str(d['city_name'].iloc[0])
    if not lats:
        return None, None, None, None
    return np.concatenate(lats), np.concatenate(lons), country, (name or city)


def _haversine_km_vec(clat, clon, lat, lon):
    """Haversine from one point to an array of points. The scalar `_haversine_km` below is
    built on `math.*` and cannot take arrays."""
    p1, p2 = math.radians(clat), np.radians(lat)
    a = (np.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * np.cos(p2) * np.sin(np.radians(lon - clon) / 2) ** 2)
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def _radius_p90(lat, lon, clat, clon) -> float:
    """90th-percentile distance (km) from a city's centre — robust to a few stray points."""
    return float(np.percentile(_haversine_km_vec(clat, clon, lat, lon), 90))


def audit_city_coords() -> dict:
    """Audit every city's coordinate cloud against its airport reference point.

    Returns {'transforms': {city: transform}, 'stats': {...}}. A transform re-anchors and
    rescales a city; cities that pass the audit get none and are written through untouched.
    """
    refs = _load_airport_refs()
    cities = sorted(
        os.path.basename(f)[:-len('_attraction.csv')]
        for f in glob.glob(os.path.join(IN_ATTR, '*_attraction.csv'))
        if os.path.basename(f) not in BARCELONA_FILES
    )

    surveyed, unmatched = [], []
    for city in cities:
        lat, lon, country, name = _read_city_cloud(city)
        if lat is None or len(lat) < 2:
            continue
        clat, clon = float(np.median(lat)), float(np.median(lon))
        ref = refs.get((_norm_key(name), _norm_key(country)))
        if not ref:
            unmatched.append(city)
            continue
        # A city with several airports takes whichever sits nearest its cloud.
        ap_dist, ap = min(
            ((_haversine_km(clat, clon, a[0], a[1]), a) for a in ref), key=lambda t: t[0])
        surveyed.append({
            'city': city, 'clat': clat, 'clon': clon,
            'radius': _radius_p90(lat, lon, clat, clon),
            'ap_dist': float(ap_dist), 'ap': ap,
        })

    displaced = [c for c in surveyed if c['ap_dist'] > DISPLACED_KM]
    dispersed = [c for c in surveyed if c['ap_dist'] <= DISPLACED_KM and c['radius'] > DISPERSED_KM]
    failing = {c['city'] for c in displaced} | {c['city'] for c in dispersed}
    passing = [c for c in surveyed if c['city'] not in failing]

    # Repaired cities inherit their dispersion from the passing cities' own distribution
    # rather than a hand-picked constant, so no scale signature separates the two groups.
    donor = np.array([c['radius'] for c in passing], dtype=float)
    coord_rng = np.random.default_rng(RNG_SEED + 1)   # separate stream: leaves H4/A1/A5 bit-identical

    transforms = {}
    for c in sorted(displaced + dispersed, key=lambda c: c['city']):
        target = float(coord_rng.choice(donor))
        # A cloud collapsed to a few hundred metres (Ankara) needs a large scale-up; guard the
        # degenerate case where every point coincides and there is no structure to scale.
        scale = target / c['radius'] if c['radius'] > 1e-6 else 1.0
        transforms[c['city']] = {
            'clat': c['clat'], 'clon': c['clon'],
            'tlat': c['ap'][0], 'tlon': c['ap'][1],
            'scale': float(scale),
            'reason': 'displaced' if c['ap_dist'] > DISPLACED_KM else 'dispersed',
            'was_radius': c['radius'], 'was_ap_dist': c['ap_dist'], 'target_radius': target,
        }

    return {
        'transforms': transforms,
        'stats': {
            'surveyed': len(surveyed), 'passing': len(passing),
            'displaced': len(displaced), 'dispersed': len(dispersed),
            'unmatched': unmatched,
            'donor_median_radius': float(np.median(donor)) if len(donor) else 0.0,
        },
    }


def apply_coord_transform(d: pd.DataFrame, tf: dict) -> int:
    """Re-anchor and rescale one city's coordinates in place. Returns rows touched."""
    if tf is None or 'latitude' not in d.columns or 'longitude' not in d.columns:
        return 0
    lat = d['latitude'].to_numpy(dtype=float)
    lon = d['longitude'].to_numpy(dtype=float)
    # Offsets in km on a local tangent plane, scaled, then re-projected about the new anchor.
    dy = (lat - tf['clat']) * DEG_KM * tf['scale']
    dx = (lon - tf['clon']) * DEG_KM * math.cos(math.radians(tf['clat'])) * tf['scale']
    d['latitude'] = tf['tlat'] + dy / DEG_KM
    d['longitude'] = tf['tlon'] + dx / (DEG_KM * math.cos(math.radians(tf['tlat'])))
    return len(d)


# ----------------------------------------------------------------------------
# Bug C3 — hotel rate_of_restaurant pegged to 5
# ----------------------------------------------------------------------------
# rate_of_restaurant is a 1-5 quality score and it was meant to rise with the hotel's star: real
# guides do rate 4- and 5-star hotel restaurants higher. But the v1 column has two breaks. Every
# star from 2.5 through 4.0 lands at ~65% fives -- the whole mid-market (76% of hotels) is
# indistinguishable -- and 4.5- and 5-star are pegged at 100% fives, every single one. Overall 66.6%
# score a perfect 5, against a real-world restaurant distribution that centres on 3.5-4.5 with
# perfect scores a minority. D1/D2's foodie persona thresholds at >=4.0 (implicit_scoring.py), and
# 328 of 1,493 queries ask for it, so two thirds of all hotels clear the bar for free.
#
# Regenerate from the intent, not from scratch: each star has a target mean rising linearly with
# star, and each hotel draws a 1-5 score from a normal around that mean, rounded and clipped. The
# star->restaurant signal the design wanted is kept (corr ~0.33, as before), the mid-market spreads
# back out, and the ceiling breaks -- even 5-star hotels now have the occasional 3. The targets are
# calibrated so 5-star hotels average exactly 4.3, the anchor the paper cites from its web survey.
# Draws come from a dedicated seed+4 stream so every other column stays bit-identical.
ROR_SIGMA = 0.9
ROR_MEAN_1STAR = 2.871    # calibrated so 1-star hotels average ~2.9 after round+clip
ROR_MEAN_5STAR = 4.429    # calibrated so 5-star hotels average exactly 4.30 after round+clip


# ----------------------------------------------------------------------------
# Bug AG — 23 cities use a second amenities_group vocabulary
# ----------------------------------------------------------------------------
# 352 cities tag personas in the canonical lowercase form the queries use ('with pets',
# 'luxury travelers'); 23 East-Asian cities carry a Title-Case, reworded batch ('Pet Owners',
# 'Luxury Travelers', 'Family with children') that appears in no other city and matches neither
# the query vocabulary nor hotel_group_facility.json. It contradicts the paper's "normalized via
# deterministic scripts" and is a visible seam on download. Fold the variants back to canonical so
# amenities_group has one vocabulary KB-wide. Deterministic string map, no RNG; the embeddings are
# regenerated downstream from the normalized text.
AMENITIES_GROUP_CANON = {
    'pet owners': 'with pets',
    'family with children': 'with children',
    'solo woman travelers': 'solo women',
    'luxury travelers': 'luxury travelers',
    'business travelers': 'business travelers',
    'elderly travelers': 'elderly travelers',
    'disabled travelers': 'disabled traveler',
    'nightlife enthusiast': 'nightlife enthusiast',
    'road trip': 'road trip',
    'couples trip': 'couples trip',
    'photography': 'photography',
    'fast-paced budget travel': 'fast-paced budget travel',
}


# ----------------------------------------------------------------------------
# Bug AR — amenities list inconsistent about the restaurant
# ----------------------------------------------------------------------------
# Every hotel has a rate_of_restaurant score and an about line "the on-site restaurant is rated
# X/5", so every hotel HAS a restaurant -- but the amenities list only names 'restaurant' on 7.5%
# of rows, all in the 23 variant-vocab cities, alongside 'Michelin-starred restaurants' (829 rows,
# all 4.5-5 star, again only those 23 cities). Two fixes: drop the Michelin token, a non-universal
# artefact of that external batch, and add 'restaurant' to every hotel so the amenities list agrees
# with the score column and the description. 'restaurant' is a universal facility like Wi-Fi, not a
# persona-conditional one (no amenities_group maps to it), so it joins the six base facilities that
# open every list rather than hanging off a group. amenities_group and its embeddings are untouched.
AR_DROP = {'Michelin-starred restaurants'}
AR_BASE_FACILITY = 'Restaurant'   # Title-Case like every other amenity (lowercase 'restaurant' was
                                  # the one exception and read as a defect)
# The six facilities every hotel already lists, in their fixed order; 'restaurant' slots in after.
AR_BASE_PREFIX = ['Wi-Fi', 'Luggage storage', 'Air conditioning',
                  'Room cleaning service', 'Safety deposit box', 'Television']


def normalize_amenities(cell):
    """Drop the Michelin outlier and guarantee a single 'restaurant' entry, placed with the base
    facilities. Order is otherwise preserved (embeddings read amenities_group, not this)."""
    try:
        items = list(ast.literal_eval(str(cell)))
    except (ValueError, SyntaxError):
        return cell, 0
    before = list(items)
    # Drop the Michelin outlier and any existing restaurant token (case-insensitively, to catch
    # v1's lowercase 'restaurant') before re-inserting the canonical one.
    items = [t for t in items if t not in AR_DROP and t.lower() != AR_BASE_FACILITY.lower()]
    # Insert 'restaurant' right after whatever prefix of the six base facilities this row starts
    # with, so it sits among the universal facilities instead of at a random tail position.
    k = 0
    while k < len(items) and k < len(AR_BASE_PREFIX) and items[k] == AR_BASE_PREFIX[k]:
        k += 1
    items.insert(k, AR_BASE_FACILITY)
    return str(items), (1 if items != before else 0)


def normalize_amenities_group(cell, unmapped: set):
    """Fold one amenities_group list to the canonical lowercase vocabulary, preserving order and
    de-duplicating any collision the fold creates."""
    try:
        items = ast.literal_eval(str(cell))
    except (ValueError, SyntaxError):
        return cell, 0
    out, seen, changed = [], set(), 0
    for t in items:
        key = str(t).strip().lower()
        canon = AMENITIES_GROUP_CANON.get(key, key)
        if canon != str(t):
            changed += 1
        if canon not in AMENITIES_GROUP_CANON.values() and key not in AMENITIES_GROUP_CANON:
            unmapped.add(str(t))
        if canon not in seen:
            seen.add(canon)
            out.append(canon)
    return str(out), (1 if changed else 0)


def regen_restaurant_rating(star_col, rng: np.random.Generator):
    """A 1-5 restaurant score per hotel, mean rising with star, 5-star anchored at 4.3."""
    star = pd.to_numeric(star_col, errors='coerce').to_numpy()
    frac = np.clip((star - 1.0) / (5.0 - 1.0), 0.0, 1.0)
    mu = ROR_MEAN_1STAR + (ROR_MEAN_5STAR - ROR_MEAN_1STAR) * frac
    draw = rng.normal(mu, ROR_SIGMA)
    return np.clip(np.round(draw), 1, 5).astype(int)


def process_hotel_file(in_path: str, out_path: str, rng: np.random.Generator,
                       tf: dict = None, ror_rng: np.random.Generator = None) -> dict:
    d = pd.read_csv(in_path)
    audit = {'records': len(d), 'h1_fixed': 0, 'h2_fixed': 0, 'h4_regen': 0, 'c1_coords': 0,
             'c3_ror': 0, 'n1_case': 0}

    # C1: re-anchor / rescale this city's coordinates if it failed the audit
    audit['c1_coords'] = apply_coord_transform(d, tf)

    # H1: rating denominator /5 -> /10
    before = d['about'].astype(str)
    d['about'] = before.apply(fix_rating_denominator)
    audit['h1_fixed'] = int((before != d['about']).sum())

    # H2: half-star truncation
    def _h2_row(row):
        return fix_star_truncation(row['about'], float(row['star']))
    before = d['about'].astype(str).copy()
    d['about'] = d.apply(_h2_row, axis=1)
    audit['h2_fixed'] = int((before != d['about']).sum())

    # H4: address regen
    d['address'] = d.apply(
        lambda r: regen_address(str(r['country']), str(r['city_name']), rng),
        axis=1,
    )
    audit['h4_regen'] = len(d)

    # C3: restaurant rating regenerated from star (own stream — leaves all else bit-identical)
    if ror_rng is not None and 'rate_of_restaurant' in d.columns and 'star' in d.columns:
        d['rate_of_restaurant'] = regen_restaurant_rating(d['star'], ror_rng)
        audit['c3_ror'] = len(d)
        # The about text ends with "the on-site restaurant is rated X/5" — that number restates the
        # column and must move with it, or 73% of rows contradict their own description (the same
        # column-vs-prose drift as A5). Rewrite it per row from the regenerated value; the /5 scale
        # is genuine and stays.
        if 'about' in d.columns:
            _PAT_REST = re.compile(r'(restaurant is rated )\d+(/5)')
            d['about'] = [
                _PAT_REST.sub(rf'\g<1>{int(v)}\g<2>', str(ab))
                for ab, v in zip(d['about'], d['rate_of_restaurant'])
            ]

    # LR: the template unconditionally says "Enjoy comfortable stays with a rating of X/10", which
    # reads as self-contradictory on the 195 hotels rated below 4/10 (a 1.0/10 hotel is not a
    # comfortable stay). Drop the "comfortable" claim to a neutral phrasing when the rating is low;
    # the number is untouched and still matches the column.
    if 'about' in d.columns and 'rating' in d.columns:
        _PAT_COMF = re.compile(r'Enjoy comfortable stays with a rating of ([\d.]+)/10')
        lr = 0
        new_about = []
        for ab, rt in zip(d['about'], d['rating']):
            s = str(ab)
            if pd.notna(rt) and float(rt) < 4.0:
                s2 = _PAT_COMF.sub(r'Rated \1/10 by guests', s)
                if s2 != s:
                    lr += 1
                s = s2
            new_about.append(s)
        d['about'] = new_about
        audit['lr_lowrating'] = lr

    # AG: fold variant amenities_group vocabulary to canonical
    if 'amenities_group' in d.columns:
        unmapped = set()
        changed = 0
        new_col = []
        for cell in d['amenities_group']:
            s, ch = normalize_amenities_group(cell, unmapped)
            new_col.append(s)
            changed += ch
        d['amenities_group'] = new_col
        audit['ag_normalized'] = changed
        audit['ag_unmapped'] = unmapped

    # AR: drop the Michelin outlier, give every hotel a 'restaurant' facility (see above). The
    # about text lists the amenities too ("...offering A, B, C. Average price..."), so it must be
    # rewritten from the new column, or 100% of rows disagree with their own description (the same
    # column-vs-prose drift as C3/A5).
    if 'amenities' in d.columns:
        new_col, ch = [], 0
        abouts = list(d['about']) if 'about' in d.columns else None
        for i, cell in enumerate(d['amenities']):
            s, c = normalize_amenities(cell)
            new_col.append(s); ch += c
            if abouts is not None:
                joined = ", ".join(ast.literal_eval(s))
                abouts[i] = re.sub(r'(offering ).+?(\. Average price)',
                                   lambda m: m.group(1) + joined + m.group(2),
                                   str(abouts[i]), count=1)
        d['amenities'] = new_col
        if abouts is not None:
            d['about'] = abouts
        audit['ar_amenities'] = ch

    # ND: fold adjacent duplicate words in hotel names ("Marrakech Inn Inn 23" -> "Marrakech Inn
    # 23"). 1,199 names carry a doubled type token (Resort Resort / Inn Inn / Suites Suites), which
    # opens the about sentence and reads as a scrape artefact. about starts with {name}, so rewrite
    # its prefix too. Runs before N1 so any collision the fold creates is caught by N1's dedup.
    if 'name' in d.columns:
        nd = 0
        for i in d.index:
            old = str(d.at[i, 'name'])
            new = re.sub(r'\b(\w+)(\s+\1\b)+', r'\1', old)
            if new != old:
                d.at[i, 'name'] = new
                nd += 1
                if 'about' in d.columns and str(d.at[i, 'about']).startswith(old):
                    d.at[i, 'about'] = new + str(d.at[i, 'about'])[len(old):]
        audit['nd_name'] = nd

    # N1: two genuinely different hotels whose names differ only in case collapse under the
    # scorer's .str.lower() matching (San Antonio's "Casa del Sol" vs "Casa Del Sol"), which folds
    # them into one entity for D0-src/D3/D6. Disambiguate the later duplicate with a suffix so
    # (name, city) is unique case-insensitively, and carry the same edit into the about text so the
    # name/about invariant this build maintains everywhere is not broken.
    if 'name' in d.columns:
        seen = {}
        for i in d.index:
            old = str(d.at[i, 'name'])
            key = old.lower()
            if key in seen:
                seen[key] += 1
                new = f"{old} ({seen[key]})"
                d.at[i, 'name'] = new
                if 'about' in d.columns:
                    ab = str(d.at[i, 'about'])
                    if ab.startswith(old):
                        d.at[i, 'about'] = new + ab[len(old):]
                audit['n1_case'] += 1
            else:
                seen[key] = 1

    d.to_csv(out_path, index=False)
    return audit


# ----------------------------------------------------------------------------
# Bug AF — attraction facilities / facilities_group variant vocabulary (= hotel's AG, unfixed here)
# ----------------------------------------------------------------------------
# The same 23 East-Asian cities that carried a Title-Case persona batch in hotels do so in
# attractions too, and it was never folded. facilities_group has 10 variant persona tokens
# ('Pet Owners' vs canonical 'with pets', etc., per attraction_group_facility.json) over 3,082
# rows; facilities has 6 lowercase-twinned tokens ('pet-friendly' vs 'Pet-friendly', etc.) over
# 13,914 rows. Fold both to canonical, feeding D1/D2's persona→facility Titan retrieval one
# vocabulary KB-wide. Deterministic, no RNG.
#
# facilities needs de-duplication after folding: 'Luggage Storage' (all rows) and 'luggage storage'
# (10,982 rows) coexist on the same row, so a plain replace would double it — unlike the hotel case.
AF_GROUP_CANON = {
    'road trip': 'road trip', 'pet owners': 'with pets',
    'fast-paced budget travel': 'fast-paced budget travel', 'solo woman travelers': 'solo women',
    'luxury travelers': 'luxury travelers', 'photography': 'photography',
    'family with children': 'with children', 'elderly travelers': 'elderly travelers',
    'nightlife enthusiast': 'nightlife enthusiast', 'disabled travelers': 'disabled traveler',
}
AF_FACILITY_CANON = {
    'luggage storage': 'Luggage Storage', 'food markets': 'Food markets',
    'nursing facilities': 'Nursing facilities', 'high speed wifi': 'High speed WiFi',
    'pet-friendly': 'Pet-friendly', 'pet rest area': 'Pet rest area',
}


def _fold_list(cell, canon_map):
    """Fold tokens via canon_map (keyed on lowercased token), preserving order, de-duping."""
    try:
        items = ast.literal_eval(str(cell))
    except (ValueError, SyntaxError):
        return cell, 0
    out, seen, changed = [], set(), 0
    for t in items:
        c = canon_map.get(str(t).strip().lower(), t)
        if c != t:
            changed += 1
        if c not in seen:
            seen.add(c)
            out.append(c)
    return str(out), (1 if changed else 0)


# Attraction ticket prices for these two types are a single global menu shuffled across every
# city (Historic: 341 cities share one 306-value price set; Museum: 337 share one 256-value set;
# sorted-per-type price vectors are byte-identical across cities). Nature and Theme Park are
# already per-city, untouched. Scale each city's within-type prices by its cost level (the same
# hotel-median multiplier used for cars) plus jitter, so no two cities share a sorted vector.
AP_CLONED_TYPES = {'Historic Site/Landmark', 'Museum/Art Gallery'}
AP_JITTER = 0.10   # +/-10%, breaks residual ties between same-multiplier cities

# AC: real attraction tickets rarely exceed these per-type caps; the KB's long tail runs to $1,510
# (a park) and $816 (a landmark), 2.7% over $200, which is unrealistic. Prices above the cap are
# not clipped (that would pile a spike at the cap) but resampled from that type's own real prices
# in the [P60, cap] band — they land back in the genuine high-price range, no spike, median
# unchanged. Runs as a post-pass over the whole domain AFTER AP, so it sees final per-city prices.
AC_CAPS = {
    'Museum/Art Gallery': 60.0, 'Historic Site/Landmark': 60.0,
    'Nature/Scenery/Park': 80.0, 'Theme Park/Amusement Park': 150.0,
}


def process_attraction_file(in_path: str, out_path: str, rng: np.random.Generator,
                            tf: dict = None, mult: float = None,
                            price_rng: np.random.Generator = None) -> dict:
    d = pd.read_csv(in_path)

    # AO: drop the 13 open_hours outliers KB-wide (7 'Always Open', 6 two-segment kaiseki
    # lunch+dinner) so every attraction is a single time window. They are real but anomalous, none
    # is referenced by any query (0 GT impact), and dropping rows changes row count — so the
    # facilities_group embeddings MUST be regenerated after this build. reset_index keeps the
    # remaining rows contiguous for the position-aligned embeddings.
    _oh = d['open_hours'].astype(str)
    ao_mask = (_oh == 'Always Open') | _oh.str.contains(',', regex=False)
    ao_dropped = int(ao_mask.sum())
    if ao_dropped:
        d = d[~ao_mask].reset_index(drop=True)

    audit = {'records': len(d), 'a1_regen': 0, 'a3_fixed': 0, 'a5_price_fixed': 0, 'c1_coords': 0,
             'ao_dropped': ao_dropped}

    # C1: same transform as this city's hotels, so both domains stay in one frame
    audit['c1_coords'] = apply_coord_transform(d, tf)

    # A1: address regen
    d['address'] = d.apply(
        lambda r: regen_address(str(r['country']), str(r['city_name']), rng),
        axis=1,
    )
    audit['a1_regen'] = len(d)

    # A3: open_hours overview fix (false alarm — kept for audit completeness, returns unchanged)
    before = d['overview'].astype(str).copy()
    d['overview'] = d.apply(
        lambda r: fix_attraction_open_hours(str(r['overview']), str(r['open_hours'])),
        axis=1,
    )
    audit['a3_fixed'] = int((before != d['overview']).sum())

    # A5 (was A6): fix attractions with "pegged" outlier ticket prices that the original
    # generation script accidentally set to two specific high values for entire type pools:
    #   - $1,654.66 for Theme Park/Amusement Park (this is the "Lotte World" example zDxD called out)
    #   - $2,059.20 for Nature/Scenery/Park
    # Replace with same-type same-city median * U(0.8, 1.2) noise; if fewer than 3 same-type
    # peers in this city, fall back to per-type defaults.
    PEGGED = {1654.66, 2059.20}
    TYPE_DEFAULT = {
        'Theme Park/Amusement Park': 22.6,
        'Nature/Scenery/Park':       12.5,
        'Historic Site/Landmark':    13.4,
        'Museum/Art Gallery':         9.5,
    }
    pegged_mask = d['ticket_price'].isin(PEGGED)
    for idx in d[pegged_mask].index:
        row = d.loc[idx]
        type_v = row['type']
        peers = d[(d['type'] == type_v) & (~d['ticket_price'].isin(PEGGED))]
        if len(peers) >= 3:
            base = peers['ticket_price'].median()
        else:
            base = TYPE_DEFAULT.get(type_v, 15.0)
        noise = 0.8 + 0.4 * float(rng.random())
        new_price = round(max(1.0, base * noise), 2)
        d.at[idx, 'ticket_price'] = new_price
        d.at[idx, 'overview'] = rewrite_overview_price(str(row['overview']), new_price)
    audit['a5_price_fixed'] = int(pegged_mask.sum())

    # AP: break the cross-city price clone in Historic/Museum by scaling each city's within-type
    # prices by its cost multiplier + jitter; sync overview's two price mentions. seed+5 (own
    # stream). Free attractions (0.0) stay free. Nature/Theme are per-city already, so skipped.
    if mult is not None and price_rng is not None:
        ap = 0
        ap_mask = d['type'].isin(AP_CLONED_TYPES)
        for idx in d[ap_mask].index:
            old_price = float(d.at[idx, 'ticket_price'])
            jitter = 1.0 + (price_rng.random() * 2 - 1) * AP_JITTER
            new_price = round(old_price * mult * jitter, 2)
            d.at[idx, 'ticket_price'] = new_price
            d.at[idx, 'overview'] = rewrite_overview_price(str(d.at[idx, 'overview']), new_price)
            ap += 1
        audit['ap_price'] = ap

    # AF: fold facilities_group + facilities variant vocabulary to canonical (embeddings read
    # facilities_group; regenerate them after this build)
    if 'facilities_group' in d.columns:
        col, ch = [], 0
        for cell in d['facilities_group']:
            s, c = _fold_list(cell, AF_GROUP_CANON); col.append(s); ch += c
        d['facilities_group'] = col
        audit['af_group'] = ch
    if 'facilities' in d.columns:
        col, ch = [], 0
        for cell in d['facilities']:
            s, c = _fold_list(cell, AF_FACILITY_CANON); col.append(s); ch += c
        d['facilities'] = col
        audit['af_facilities'] = ch

    # BP: the KB maps 'business travelers' persona <-> the 'High speed WiFi' facility bidirectionally
    # (every attraction tagged business has WiFi; 97% of WiFi attractions are tagged business). The
    # 23 East-Asian cities broke the reverse direction — 1,426 rows list High speed WiFi but omit
    # the persona from facilities_group, so business-travel queries mis-rank them (D1/D2). Add the
    # persona back where the facility is present and the tag is missing, restoring the mapping.
    if {'facilities', 'facilities_group'} <= set(d.columns):
        bp = 0
        col = []
        for fac, grp in zip(d['facilities'], d['facilities_group']):
            try:
                fl = ast.literal_eval(str(fac)); gl = ast.literal_eval(str(grp))
            except (ValueError, SyntaxError):
                col.append(grp); continue
            if 'High speed WiFi' in fl and 'business travelers' not in gl:
                gl = list(gl) + ['business travelers']
                bp += 1
            col.append(str(gl))
        d['facilities_group'] = col
        audit['bp_persona'] = bp

    # AV: 130 theme parks recommend an 8-hour visit but open only 09:30-17:00 (7.5h) — an
    # impossible visit that would make any itinerary using them infeasible (D5/D6). All 130 are the
    # identical case. Widen their hours to 09:30-20:30 (11h, comfortably fits 8h; parks open into
    # the evening is realistic) and sync the "Open ..." clause in overview.
    if {'type', 'open_hours', 'duration_of_visit'} <= set(d.columns):
        av_mask = ((d['type'] == 'Theme Park/Amusement Park') &
                   (d['open_hours'].astype(str).str.strip() == '09:30-17:00') &
                   (d['duration_of_visit'].astype(str).str.strip() == '8 hours'))
        for idx in d[av_mask].index:
            d.at[idx, 'open_hours'] = '09:30-20:30'
            d.at[idx, 'overview'] = str(d.at[idx, 'overview']).replace(
                'Open 09:30-17:00', 'Open 09:30-20:30')
        audit['av_hours'] = int(av_mask.sum())

    # NW: 731 attractions have open_hours whose close is <= open (08:00-00:00, 20:00-05:00, ...) —
    # real overnight venues, but scoring.py's is_time_in_range compares naively and marks them
    # always-closed. Per the decision to fix the data (not the scorer), collapse each to a single-day
    # window keeping the open time: a midnight close (00:00) becomes 23:00; a small-hours close
    # (<= open) becomes a 23:00 close, and if that leaves < 3h, pull the open to 18:00. Sync overview.
    if 'open_hours' in d.columns:
        nw = 0
        for idx in d.index:
            m = re.match(r'(\d+):(\d+)-(\d+):(\d+)$', str(d.at[idx, 'open_hours']).strip())
            if not m:
                continue
            o = int(m[1]) * 60 + int(m[2]); c = int(m[3]) * 60 + int(m[4])
            if c > o:
                continue
            open_hm = f'{m[1]}:{m[2]}'
            new = f'{open_hm}-23:00'
            if o > 20 * 60:            # opens after 20:00 -> pull open to 18:00 for a sane window
                new = '18:00-23:00'
            old_oh = str(d.at[idx, 'open_hours']).strip()
            d.at[idx, 'open_hours'] = new
            d.at[idx, 'overview'] = str(d.at[idx, 'overview']).replace(
                f'Open {old_oh}', f'Open {new}')
            nw += 1
        audit['nw_hours'] = nw

    # AD: 24 adult-only venues (Thai cabaret shows, nightclubs, VIP cocktail lounges) carry
    # 'with children' in facilities_group. That is not a scheduling nuisance but a corrupted D1
    # signal: an agent doing persona matching is told a nightclub suits a family, and is REWARDED
    # for booking it. Strip the tag; facilities_group is not mirrored in overview (verified: 2 of
    # 5,827 overviews mention any facility string), so no prose sync is needed.
    if 'facilities_group' in d.columns and 'attraction_name' in d.columns:
        adult_re = re.compile(
            r'\b(nightclub|night club|cabaret|strip club|burlesque|casino|gentlemen|adult|'
            r'brothel|red[- ]light|hostess|go[- ]go|cocktail lounge|vip lounge)\b', re.I)
        ad = 0
        for idx in d.index:
            if not adult_re.search(str(d.at[idx, 'attraction_name'])):
                continue
            try:
                tags = ast.literal_eval(str(d.at[idx, 'facilities_group']))
            except Exception:
                continue
            if 'with children' not in tags:
                continue
            d.at[idx, 'facilities_group'] = repr([t for t in tags if t != 'with children'])
            ad += 1
        audit['ad_adult_detag'] = ad

    # NH: 196 night PRODUCTS (sunset cruises, moonlight tours, night safaris) advertise daytime
    # hours -- 'Ribeira River Sunset Cruise' open 09:00-18:00. B2 then certifies a 10:00 sunset
    # cruise as compliant, and the reference planner is forced to schedule it in the morning.
    # Move each to an evening window ending 22:00, preserving its window length up to 5h, and sync
    # the 'Open X-Y' clause that every overview restates. Deterministic, no RNG.
    if 'open_hours' in d.columns and 'attraction_name' in d.columns:
        night_re = re.compile(
            r'(night\s+(safari|cruise|tour|market|show|walk|bus|sightseeing|view)|'
            r'(sunset|evening|moonlight|starlight|midnight)\s+'
            r'(cruise|sail|tour|safari|show|dinner|carnival)|after[- ]dark|night\s*life)', re.I)
        nh = 0
        for idx in d.index:
            if not night_re.search(str(d.at[idx, 'attraction_name'])):
                continue
            m = re.match(r'(\d+):(\d+)-(\d+):(\d+)$', str(d.at[idx, 'open_hours']).strip())
            if not m:
                continue
            o = int(m[1]) * 60 + int(m[2]); c = int(m[3]) * 60 + int(m[4])
            if c >= 20 * 60:
                continue
            span = min(max(c - o, 120), 5 * 60)
            new_o = max(16 * 60, 22 * 60 - span)
            new = f'{new_o // 60:02d}:{new_o % 60:02d}-22:00'
            old_oh = str(d.at[idx, 'open_hours']).strip()
            d.at[idx, 'open_hours'] = new
            d.at[idx, 'overview'] = str(d.at[idx, 'overview']).replace(
                f'Open {old_oh}', f'Open {new}')
            nh += 1
        audit['nh_night_hours'] = nh

    # CO: 2 attractions sit >80 km from their own city's median (Funchal 'Echoes of the Island'
    # 298 km out in the Atlantic; Doha 'Qatari Heritage Nexus' 101 km). Coordinates are sandbox
    # synthetic, so snap the outlier back inside the city on a DETERMINISTIC small offset (no RNG
    # stream to perturb). Coordinates are not mirrored in overview, so no prose sync.
    if {'latitude', 'longitude'}.issubset(d.columns) and len(d) > 3:
        mlat = float(pd.to_numeric(d['latitude'], errors='coerce').median())
        mlon = float(pd.to_numeric(d['longitude'], errors='coerce').median())
        co = 0
        for n, idx in enumerate(d.index):
            try:
                la = float(d.at[idx, 'latitude']); lo = float(d.at[idx, 'longitude'])
            except Exception:
                continue
            if _haversine_km(mlat, mlon, la, lo) <= 80.0:
                continue
            d.at[idx, 'latitude'] = round(mlat + ((n % 7) - 3) * 0.01, 6)
            d.at[idx, 'longitude'] = round(mlon + ((n % 5) - 2) * 0.01, 6)
            co += 1
        audit['co_coord_outlier'] = co

    # AM: 29 attraction names carry U+FFFD replacement chars where a separator failed to decode
    # (all in the 23-city batch). The original char is unrecoverable, but the intent — a separator
    # between "{City}" and the event — is clear; replace each run of U+FFFD (with its surrounding
    # spaces) with ": ", and trim a trailing one. overview starts with the name, so rewrite its
    # prefix in sync. Deterministic, no RNG.
    if 'attraction_name' in d.columns:
        am = 0
        for idx in d.index:
            old = str(d.at[idx, 'attraction_name'])
            if '�' not in old:
                continue
            def _clean(t):
                return re.sub(r':\s*$', '', re.sub(r'\s*�+\s*', ': ', t)).strip()
            new = _clean(old)
            d.at[idx, 'attraction_name'] = new
            am += 1
            # overview repeats the name twice ("{name} is a ..." and "Discover {name}, ..."), so
            # replace every U+FFFD run in the whole overview, not just the prefix. When the name
            # ended in a mojibake run, the replacement leaves a dangling ": " before the following
            # word ("Nagoya Station: is a nature"); drop that colon so the prose reads cleanly.
            if 'overview' in d.columns:
                ov = re.sub(r'\s*�+\s*', ': ', str(d.at[idx, 'overview']))
                ov = re.sub(r':\s+(is a\b|,)', r'\1', ov)
                d.at[idx, 'overview'] = ov
        audit['am_name'] = am

    # NC: same as hotels' N1 — attraction names that differ only in case (Detroit
    # "Playzone"/"PlayZone", Fuzhou "Dreamplay"/"DreamPlay") collapse under core_api's
    # .str.lower() matching into one entity for D0-src. Suffix the later duplicate so (name, city)
    # is case-insensitively unique, and carry the edit into overview (which opens with the name).
    if 'attraction_name' in d.columns:
        seen, nc = {}, 0
        for idx in d.index:
            old = str(d.at[idx, 'attraction_name'])
            key = old.lower()
            if key in seen:
                seen[key] += 1
                new = f"{old} ({seen[key]})"
                d.at[idx, 'attraction_name'] = new
                if 'overview' in d.columns and str(d.at[idx, 'overview']).startswith(old):
                    d.at[idx, 'overview'] = new + str(d.at[idx, 'overview'])[len(old):]
                nc += 1
            else:
                seen[key] = 1
        audit['nc_case'] = nc

    d.to_csv(out_path, index=False)
    return audit


# ----------------------------------------------------------------------------
# Bug CR — 26 cities whose coordinates C1 left crushed (<2 km) or dispersed (points out of city)
# ----------------------------------------------------------------------------
# C1's pooled audit+rescale left 14 cities crushed to a point (Caselle 55 m, Guangzhou 176 m — a
# city's hotel centroid was 2000 km away in Japan, so pooling blew the radius and the rescale
# collapsed the correct attractions too) and 12 dispersed (Wellington/Kaohsiung points 100-1500 km
# out of city). The other 349 cities — including the 37 C1 repaired correctly — are left untouched.
#
# For each of the 26, take the robust centroid (median, so a dispersed city's outliers are ignored)
# of that city's own points and scatter fresh points within CR_RADIUS on an equal-area disc. Hotel
# and attraction share one centroid per city, so both land in the same city (no hotel-outside-
# attractions). Coordinates are synthetic sandbox geography, so only that each point is in-city,
# distinct, and sanely dispersed matters — which this guarantees. Own seed stream (seed+7). Only
# D6 reads coordinates.
CR_CITIES = [
    'Balice', 'Caselle', 'Gibraltar', 'Guangzhou', 'Hagåtña', 'Kinshasa', 'Manaus', 'Muscat',
    'Orio al Serio', 'Perth', 'Shannon', 'Valletta', 'Vieux Fort', 'Willemstad',
    'Wellington', 'Attock', 'Kaohsiung', 'Auckland', 'Zhengzhou', 'Amman', 'Shenzhen',
    'Anchorage', 'Guiyang', 'Dubai', 'Hamburg', 'Karachi',
]
CR_RADIUS_KM = 16.0   # median in-city radius of the 355 healthy cities


def _scatter_in_city(clat, clon, n, rng):
    """n distinct (lat,lon) uniformly on a disc of CR_RADIUS_KM around (clat, clon)."""
    r = np.sqrt(rng.random(n)) * CR_RADIUS_KM      # sqrt => area-uniform, no centre pile-up
    th = rng.random(n) * 2 * np.pi
    lat = clat + (r * np.sin(th)) / DEG_KM
    lon = clon + (r * np.cos(th)) / (DEG_KM * math.cos(math.radians(clat)))
    return lat, lon


def regenerate_city_coords(rng: np.random.Generator) -> dict:
    """Regenerate hotel+attraction coordinates for the 26 CR cities, in-city, around each city's
    own robust centroid. One centroid per city shared by both domains."""
    fixed = 0
    for city in CR_CITIES:
        # Robust centroid from the city's current coordinates, pooling both domains but via the
        # median so a dispersed domain's outliers don't drag it.
        lats, lons, files = [], [], []
        for out_dir, suffix in ((OUT_HOTEL, '_hotel.csv'), (OUT_ATTR, '_attraction.csv')):
            p = os.path.join(out_dir, city + suffix)
            if os.path.exists(p):
                d = pd.read_csv(p)
                if 'latitude' in d.columns:
                    lats.append(d['latitude'].to_numpy(dtype=float))
                    lons.append(d['longitude'].to_numpy(dtype=float))
                    files.append((p, d))
        if not files:
            continue
        clat = float(np.median(np.concatenate(lats)))
        clon = float(np.median(np.concatenate(lons)))
        for p, d in files:
            nlat, nlon = _scatter_in_city(clat, clon, len(d), rng)
            d['latitude'] = nlat
            d['longitude'] = nlon
            d.to_csv(p, index=False)
            fixed += len(d)
    return {'cr_rows': fixed, 'cr_cities': len(CR_CITIES)}


def cap_attraction_prices(out_dir: str, rng: np.random.Generator) -> dict:
    """AC post-pass: after every city is written (AP done), pull the whole domain, and for each
    type resample any ticket_price above AC_CAPS[type] from that type's own real prices in the
    [P60, cap] band. Free stays free. overview's two price mentions are re-synced. Runs after AP so
    it sees final per-city prices; global so the resample pool is the real cross-city distribution.
    """
    files = sorted(glob.glob(os.path.join(out_dir, '*_attraction.csv')))
    frames = {f: pd.read_csv(f) for f in files}
    allrows = pd.concat(frames.values(), ignore_index=True)
    pools = {}
    for t, cap in AC_CAPS.items():
        p = allrows.loc[allrows['type'] == t, 'ticket_price']
        under = p[(p > 0) & (p <= cap)]
        pools[t] = (under[under >= under.quantile(0.60)].to_numpy()
                    if len(under) else np.array([cap]))
    capped = 0
    for f, d in frames.items():
        changed = False
        for idx in d.index:
            cap = AC_CAPS.get(d.at[idx, 'type'])
            if cap is not None and float(d.at[idx, 'ticket_price']) > cap:
                new_price = round(float(rng.choice(pools[d.at[idx, 'type']])), 2)
                d.at[idx, 'ticket_price'] = new_price
                d.at[idx, 'overview'] = rewrite_overview_price(str(d.at[idx, 'overview']), new_price)
                capped += 1
                changed = True
        if changed:
            d.to_csv(f, index=False)
    return {'ac_capped': capped}


# ----------------------------------------------------------------------------
# Bug C2 — 375 cities, 23 price lists
# ----------------------------------------------------------------------------
# Pivot city x car_id to price_per_day and the 375 cities collapse to 23 distinct tables, none of
# them unique to a city: Kigali and Los Angeles are equal to the cent, and one EV triple is shared
# verbatim by 18 cities across four continents. The car_type and capacity vectors are worse -- one
# distinct value each across all 375, so every city rents the identical 27-car fleet and price is
# the only column that moves at all. 10,125 rows hold 600 distinct prices.
#
# It is not a house style: the same test run on hotels and attractions gives 375 distinct tables
# and 375 singletons apiece. The pipeline meant every city to have its own prices; cars are the
# one domain where that did not happen. Nor is it "car hire is a global commodity" -- per-city
# mean car price spans 1.12x worldwide against hotel's 2.61x and attraction's 4.13x, and a
# reviewer needs one `diff` to show the car domain is 23 cities, not the 375 the abstract claims.
# The measurable cost: car_type alone predicts price to 12.8% median error, so an agent can budget
# a rental without ever querying the KB (the same trick on hotels is 39.3% off).
#
# The repair takes each city's cost level from the one price column that IS already per-city and
# grounded -- its median hotel price against the global median -- and scales that city's cars by
# it. Every car keeps its old price's share of its cell, so the type ordering (Economy < Compact <
# ... < Luxury) and the within-cell spread that tracks extra_services both survive; only the city
# level moves. This is why the paper can say prices are scaled by local cost level rather than
# resampled: the multiplier is read out of real per-city data, not invented.
CAR_PRICE_JITTER = 0.05   # +/-5%, enough to break exact ties between same-cost cities


def city_cost_multipliers() -> dict:
    """canonical city -> its cost level, as median hotel price over the global median.

    Read from v1 hotels, whose prices are untouched by this build and already per-city distinct
    (375/375 unique), so the multiplier carries real signal rather than one this script invented.
    """
    med = {}
    for f in sorted(glob.glob(os.path.join(IN_HOTEL, '*_hotel.csv'))):
        if os.path.basename(f) in BARCELONA_FILES:
            continue
        d = pd.read_csv(f)
        if 'price' not in d.columns or not len(d):
            continue
        med[canonical_city(d['city_name'].iloc[0])] = float(
            pd.to_numeric(d['price'], errors='coerce').median())
    if not med:
        return {}
    world = float(np.median(list(med.values())))
    return {c: v / world for c, v in med.items()}


def process_car_file(in_path: str, out_path: str, mult: float = None,
                     rng: np.random.Generator = None, price_map: dict = None) -> dict:
    d = pd.read_csv(in_path)
    audit = {'records': len(d), 'c2_repriced': 0}

    if mult is not None and rng is not None and 'price_per_day' in d.columns:
        city = canonical_city(d['city_name'].iloc[0])
        old = pd.to_numeric(d['price_per_day'], errors='coerce')
        jitter = 1.0 + (rng.random(len(d)) * 2 - 1) * CAR_PRICE_JITTER
        new = (old * mult * jitter).round(2)
        if price_map is not None:
            for o, n, t in zip(old, new, d['car_type']):
                price_map[(city, str(t), round(float(o), 2))] = float(n)
        d['price_per_day'] = new
        audit['c2_repriced'] = int(new.notna().sum())

    d.to_csv(out_path, index=False)
    return audit


# ----------------------------------------------------------------------------
# Bug F5 — city filenames that no consumer can reconstruct
# ----------------------------------------------------------------------------
# Every per-city file is addressed by name: the API builds `<city>_rental_cars.csv` from the
# requested city and reads it. v1's filenames cannot be rebuilt from the city_name column they
# claim to encode, three ways over 37 of 375 cities:
#
#   * an apostrophe was escaped to `_` -- `Xi'an` is stored as `Xi_an_rental_cars.csv`, though
#     apostrophes are legal filenames on every filesystem the release targets (3 cities);
#   * the name is stored NFD while the column is NFC -- `Brasília` on disk is `i` + U+0301, so a
#     byte-exact lookup of the column value misses (27 cities);
#   * core_api applies .title(), which mangles lowercase particles: `Ciudad de México` becomes
#     `Ciudad De México` (12 cities).
#
# macOS hides the last two: APFS is case- and normalisation-insensitive, so the author's machine
# resolves them and a reviewer's ext4 does not. That is the dangerous shape -- invisible here,
# certain there. When the lookup misses, `search_cars` returns an error dict, and
# data_loader.get_car_price falls through to a loop over every city that ignores its city
# argument entirely, so `get_car_price('EV', "Xi'an")` answers 116.14 -- Amsterdam's price.
#
# Fix both ends. Here: name every output file for the NFC city_name it contains, apostrophes and
# all, so filename == column value byte for byte. In core_api/data_loader: index by city_name
# instead of rebuilding a path from a .title()'d string.
def canonical_city(name) -> str:
    """The one spelling of a city: NFC, trimmed. Filenames and lookups both go through this."""
    return unicodedata.normalize('NFC', str(name)).strip()


def city_of_csv(path: str) -> str:
    """The canonical city a per-city CSV belongs to, read from its city_name column."""
    d = pd.read_csv(path, nrows=1, keep_default_na=False, na_values=[''])
    col = 'city_name' if 'city_name' in d.columns else 'city'
    return canonical_city(d[col].iloc[0])


def copy_embedding_subdir(parent_in: str, parent_out: str, subdir: str,
                          rename: dict = None) -> dict:
    """Copy a per-city .npy embedding folder, skipping Barcelona variants and any non-.npy.

    `rename` maps a v1 filename stem to its canonical city so the embeddings stay addressable by
    the same key as the CSVs -- they are looked up as `<city>_<suffix>.npy` by the same code path.
    """
    src = os.path.join(parent_in, subdir)
    dst = os.path.join(parent_out, subdir)
    if not os.path.isdir(src):
        return {'copied': 0, 'skipped_barcelona': 0, 'src_missing': True}
    os.makedirs(dst, exist_ok=True)
    copied = 0; skipped = 0; renamed = 0
    for f in os.listdir(src):
        if not f.endswith('.npy'):
            continue
        if any(f.startswith(p) for p in BARCELONA_NPY_PREFIXES):
            skipped += 1
            continue
        out_name = f
        if rename:
            # Longest stem first: 'St. John_s' must win over any shorter prefix.
            for stem in sorted(rename, key=len, reverse=True):
                if f.startswith(stem + '_'):
                    cand = rename[stem] + f[len(stem):]
                    if cand != f:
                        out_name = cand
                        renamed += 1
                    break
        shutil.copyfile(os.path.join(src, f), os.path.join(dst, out_name))
        copied += 1
    return {'copied': copied, 'skipped_barcelona': skipped, 'src_missing': False,
            'renamed': renamed}


def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def _add_minutes_hhmm(time_str, mins):
    h, m = map(int, str(time_str).split(':'))
    total = (h * 60 + m + mins) % (24 * 60)
    return f"{total//60:02d}:{total%60:02d}"


def load_route_durations(in_dir: str) -> dict:
    """(dep_iata, arr_iata) -> median real duration in minutes, from the source snapshot.

    agoda_direct_20250801_merged_FINAL.csv is the snapshot flights.csv was built from, and it
    still carries what flights.csv threw away: full timestamps (`2025-08-01T10:30`), not bare
    HH:MM. That dropped date is why a duration is ambiguous here at all -- an overnight leg and
    a one-minute leg look identical once the date is gone.

    The snapshot cannot be joined row-wise: anonymisation regenerated flight numbers and clock
    times, so a shared flight_id means nothing (departure stamps agree on 0.1% of rows). What
    survives is the route: KB and snapshot durations agree at rho=0.996 per route, median ratio
    1.001. So a repaired flight takes its route's real median duration, which lands it on the
    same distribution as the 98.6% of flights that were already sound. The Haversine estimate
    that would otherwise be used runs ~8% short, and would leave repaired flights detectably
    faster than their neighbours.

    The snapshot has its own scraping artifacts (~5% of rows compute to a non-positive
    duration), so rows outside a sane band are dropped before taking the median.
    """
    src = os.path.join(in_dir, 'agoda_direct_20250801_merged_FINAL.csv')
    if not os.path.exists(src):
        return {}
    a = pd.read_csv(src)
    dep = pd.to_datetime(a['departureTime'], errors='coerce')
    arr = pd.to_datetime(a['arrivalTime'], errors='coerce')
    od = a['departureAirport'].map(lambda i: AIRPORT_TZ.get(i, (None,))[DST_SUM] if i in AIRPORT_TZ else None)
    oa = a['arrivalAirport'].map(lambda i: AIRPORT_TZ.get(i, (None,))[DST_SUM] if i in AIRPORT_TZ else None)
    dur = (arr - dep).dt.total_seconds() / 60 - (oa - od)
    ok = dur.notna() & dur.between(15, 1200)
    g = a[ok].assign(_d=dur[ok]).groupby(['departureAirport', 'arrivalAirport'])['_d'].median()
    return {k: float(v) for k, v in g.items()}


def _flight_duration_min(row, dst: int) -> float:
    """Real duration in minutes under one DST interpretation, resolving the missing date by
    taking the flight to land within 24h of departure."""
    dep = _hhmm_to_min(row['departure_time'])
    arr = _hhmm_to_min(row['arrival_time'])
    tz_d = AIRPORT_TZ[row['departure_airport_iata_code']][dst]
    tz_a = AIRPORT_TZ[row['arrival_airport_iata_code']][dst]
    return ((arr - dep) - (tz_a - tz_d)) % (24 * 60)


def _hhmm_to_min(t) -> int:
    h, m = map(int, str(t).split(':'))
    return h * 60 + m


def fix_zero_duration_flights(df: pd.DataFrame, route_dur: dict = None) -> dict:
    """Bug F1: flight times that no timezone reading can reconcile with the distance flown.

    The previous implementation inferred timezones from the country code and bailed out on
    anything cross-border or in a country it listed as multi-timezone. That let two families
    through. It skipped every domestic Chinese flight -- China was on the multi-timezone list,
    yet the CAAC schedules the whole country in Beijing time -- and it accepted any flight whose
    arrival stamp merely came after its departure stamp, never asking whether the gap was long
    enough to fly the distance. Together those left "Chongqing -> Shanghai, 1,460 km, 16:02 ->
    16:03" and "Newark -> Mumbai, 12,546 km, 11:45 -> 11:48" in the release.

    This version reads each airport's true offset from AIRPORT_TZ and tests the duration against
    the Haversine estimate in both directions -- too fast to be possible, or (after the midnight
    wrap) so slow it is really a negative duration in disguise.

    The offset used is AIRPORT_TZ's daylight column, which is the one actually in force on the
    snapshot date (verified against IANA at 2025-08-01 for all 392 airports). An earlier version
    spared any flight that was plausible under EITHER column, meaning to avoid rewriting anything
    on a DST guess. There is no guess to avoid: the snapshot is dated, and load_route_durations
    already treats the daylight column as the only truth when it reads that same snapshot. The
    two-column rule only reprieved 375 flights via a January reading of an August day -- among
    them PHX->LAS 15:50->15:50, which Arizona's lack of DST makes a zero-minute flight in summer.
    Healthy flights sit at 0.73-2.08x the estimate (P1-P99), well inside the [0.6, 3.0] band.

    Repairs rewrite arrival_time to departure + the route's real duration + the offset
    difference, using the snapshot-date (daylight) offset, and keep the arrival stamp local to
    its airport. Detection still uses the Haversine estimate, which needs no snapshot and is
    only ever asked whether a duration is off by more than a factor -- the repair is where the
    8% bias would matter, so that is where the real duration is used. Routes missing from the
    snapshot (0.1%) fall back to the estimate.
    """
    FAST, SLOW = 0.6, 3.0
    route_dur = route_dur or {}

    def _needed(row):
        dist = _haversine_km(row['departure_airport_latitude'], row['departure_airport_longitude'],
                             row['arrival_airport_latitude'], row['arrival_airport_longitude'])
        return max(30.0, dist / 800 * 60 + 30)

    fixed_fast = fixed_slow = 0
    unknown_airport = 0
    from_snapshot = from_estimate = 0
    for idx, row in df.iterrows():
        if (row['departure_airport_iata_code'] not in AIRPORT_TZ
                or row['arrival_airport_iata_code'] not in AIRPORT_TZ):
            unknown_airport += 1
            continue

        need = _needed(row)
        dur = _flight_duration_min(row, DST_SUM)
        if need * FAST <= dur <= need * SLOW:
            continue

        if dur < need * FAST:
            fixed_fast += 1
        else:
            fixed_slow += 1

        # Trust the snapshot, but not blindly: 7 of its 11,481 routes carry a median that is
        # itself impossible (CKG->XIY, 562 km, median 438 min -- a one-hour hop). Feeding those
        # back in would repair a flight into a fresh violation, so a route median has to clear
        # the same physical check the flights do.
        real = route_dur.get((row['departure_airport_iata_code'], row['arrival_airport_iata_code']))
        if real is not None and need * FAST <= real <= need * SLOW:
            new_dur = real
            from_snapshot += 1
        else:
            new_dur = need
            from_estimate += 1

        tz_d = AIRPORT_TZ[row['departure_airport_iata_code']][DST_SUM]
        tz_a = AIRPORT_TZ[row['arrival_airport_iata_code']][DST_SUM]
        shift = int(round(new_dur)) + (tz_a - tz_d)
        df.at[idx, 'arrival_time'] = _add_minutes_hhmm(row['departure_time'], shift)

    return {'fixed_zero': fixed_fast, 'fixed_negative': fixed_slow,
            'fixed_total': fixed_fast + fixed_slow, 'unknown_airport': unknown_airport,
            'from_snapshot': from_snapshot, 'from_estimate': from_estimate}


def rejoin_airport_metadata(df: pd.DataFrame, airports: pd.DataFrame) -> dict:
    """Bug F3: re-derive each flight's airport metadata from VALID_airports_id.csv by IATA.

    flights.csv carries airport name, city and coordinates denormalised from the scrape, and the
    scrape disagrees with the project's own airport table in three ways:

      * NRT is labelled `Tokyo` on 262 rows and `Narita` on 1,018 -- the only IATA of 392 with
        two city labels. The API matches city by exact lowercase string (core_api.py), so the
        split hides 35 of Narita's destinations from anyone asking for Tokyo, and hides cheaper
        seats on the same aircraft behind the other label (Tokyo->San Francisco: $1,787 under
        `Tokyo`, $376 under `Narita`). The table says Narita, and so does the KB's own naming
        convention -- Zaventem, Otopeni, Balice, Caselle and a dozen more are all named for the
        airport's town and carry their own hotel/attraction/car files.
      * TAO is named `Liuting Airport` on 259 rows -- an airport that closed in August 2021, four
        years before this snapshot. Their coordinates are Jiaodong's to within 971 m, so it is a
        stale name on the right airport, not a second one. The agent is shown airport_name but
        not IATA, so it has no way to tell the two names are one place.
      * Six airports carry coordinates that disagree with the table: PKX by 1,449 m (truncated to
        one decimal, and truncated *consistently*, so no duplicate-value check could see it),
        CTU by 2,253 m, CKG, TAO, HGH, SHA. D6 reads these coordinates.

    Taking the table as authoritative for name/city/coordinates/continent fixes all three at
    once. NKM and OKD are not in the table -- it lists only large_airport rows and those two are
    not -- so they keep their scraped values and are backfilled by fix_airport_country_nulls.
    """
    v = airports.dropna(subset=['iata_code']).drop_duplicates('iata_code').set_index('iata_code')
    stats = defaultdict(int)
    for side in ('departure', 'arrival'):
        iata = df[f'{side}_airport_iata_code']
        known = iata.isin(v.index)
        for col, src in ((f'{side}_airport_name', 'airport_name'),
                         (f'{side}_city', 'city'),
                         (f'{side}_airport_continent', 'continent')):
            new = iata.map(v[src])
            changed = known & (df[col].astype(str) != new.astype(str))
            stats[col] += int(changed.sum())
            df.loc[changed, col] = new[changed]
        for col, src in ((f'{side}_airport_latitude', 'latitude_deg'),
                         (f'{side}_airport_longitude', 'longitude_deg')):
            new = pd.to_numeric(iata.map(v[src]), errors='coerce')
            cur = pd.to_numeric(df[col], errors='coerce')
            changed = known & ((cur - new).abs() > 5e-4)
            stats[col] += int(changed.sum())
            df.loc[known, col] = new[known]
    return dict(stats)


def fix_zero_flight_numbers(df: pd.DataFrame, rng: np.random.Generator) -> dict:
    """Bug F4: 20 flights are numbered `XX 0000`. No airline has a flight 0.

    Flight numbers here are synthetic by design and disclosed as such (Appendix: "replaced with
    randomly generated codes following the IATA two-letter carrier prefix format"), which is the
    right call -- a structure-preserving pseudonym would let the route network re-identify the
    real carrier. Randomness is not the defect; drawing zero is. The generator sampled 0-9999 and
    zero-padded to four digits, so 0000 came up 20 times.

    Only those 20 are redrawn, from 1-9999 in the same four-digit format, keeping the numbers
    globally unique (they are the D0-src key, matched as an opaque string). No query references a
    flight number, so no annotation moves.
    """
    zero = df['flight_number'].str.match(r'^[A-Z]{2} 0+$', na=False)
    taken = set(df['flight_number'])
    fixed = 0
    for idx in df.index[zero]:
        prefix = df.at[idx, 'flight_number'].split()[0]
        while True:
            cand = f'{prefix} {int(rng.integers(1, 10000)):04d}'
            if cand not in taken:
                break
        taken.add(cand)
        df.at[idx, 'flight_number'] = cand
        fixed += 1
    return {'fixed': fixed}


def fix_airport_country_nulls(df: pd.DataFrame, airports: pd.DataFrame) -> dict:
    """Bug F2: 401 flights carry no country/continent for their airport.

    Not missing data -- a failed join. flights.csv speaks ISO-2 ('JP'), VALID_airports_id.csv
    speaks full names ('Japan'), so whatever built flights.csv could not reconcile the two and
    left three Japanese airports blank: OKA (Okinawa, 391 rows), NKM (Nagoya, 10), OKD
    (Sapporo, 4). Every other column on those rows is intact.

    The ISO-2 mapping is learned from the data rather than hardcoded: airports whose flights DO
    carry a country code pin their table country name to that code, which resolves 134 of 135
    countries with no ambiguity. NKM and OKD are absent from the airport table entirely, so they
    fall back to their city, whose other airport (NGO, CTS) supplies the same country.
    """
    name_to_iso, name_to_cont = {}, {}
    for side in ('departure', 'arrival'):
        j = df[[f'{side}_airport_iata_code', f'{side}_airport_country',
                f'{side}_airport_continent']].dropna()
        j.columns = ['iata_code', 'iso2', 'continent']
        m = airports[['iata_code', 'country']].merge(j, on='iata_code')
        for country, grp in m.groupby('country'):
            name_to_iso.setdefault(country, grp['iso2'].mode().iloc[0])
            name_to_cont.setdefault(country, grp['continent'].mode().iloc[0])

    by_iata = airports.dropna(subset=['iata_code']).drop_duplicates('iata_code').set_index('iata_code')
    by_city = airports.dropna(subset=['city']).drop_duplicates('city').set_index('city')

    def resolve(iata, city):
        row = by_iata.loc[iata] if iata in by_iata.index else (
            by_city.loc[city] if city in by_city.index else None)
        if row is None:
            return None, None
        country = row['country']
        return name_to_iso.get(country), name_to_cont.get(country, row.get('continent'))

    filled = 0
    unresolved = set()
    for side in ('departure', 'arrival'):
        co, ct = f'{side}_airport_country', f'{side}_airport_continent'
        gap = df[co].isna() | df[ct].isna()
        for idx in df.index[gap]:
            iso, cont = resolve(df.at[idx, f'{side}_airport_iata_code'], df.at[idx, f'{side}_city'])
            if iso is None:
                unresolved.add(df.at[idx, f'{side}_airport_iata_code'])
                continue
            df.at[idx, co], df.at[idx, ct] = iso, cont
            filled += 1

    return {'filled': filled, 'unresolved': sorted(unresolved)}


def process_flights(in_dir: str, out_dir: str) -> dict:
    """Build the release flight tables: drop the Barcelona airports, backfill the failed
    country join, and rewrite the flight times no timezone can explain."""
    os.makedirs(out_dir, exist_ok=True)
    # keep_default_na=False, or pandas reads the continent of North America -- the string 'NA' --
    # as a null. Whatever built these files hit that and worked around it by embedding literal
    # quote characters in the value, so the released continent is the 4-character `"NA"` rather
    # than `NA`, and `df[df.continent == 'NA']` matches nothing. Read the quotes in, strip them,
    # write the bare code back out. (Downstream readers using pandas defaults will see NaN for
    # North America; nothing reads continent -- it is absent from the API's columns_to_keep and
    # unreferenced by scoring -- so the file being right is what matters here.)
    src = os.path.join(in_dir, 'flights_no_barcelona.csv')
    df = pd.read_csv(src, keep_default_na=False, na_values=[''])

    airports = pd.read_csv(os.path.join(in_dir, 'VALID_airports_id.csv'),
                           keep_default_na=False, na_values=[''])
    for frame, cols in ((df, ['departure_airport_continent', 'arrival_airport_continent']),
                        (airports, ['continent'])):
        for c in cols:
            frame[c] = frame[c].astype(str).str.strip('"')
    # H3 removed both Barcelonas from every other table but left their airports advertised
    # here, so the release still listed two cities you could neither fly to nor book in.
    n_airports_in = len(airports)
    airports = airports[~airports['iata_code'].isin(BARCELONA_AIRPORTS)].copy()
    still_used = df['departure_airport_iata_code'].isin(BARCELONA_AIRPORTS) | \
        df['arrival_airport_iata_code'].isin(BARCELONA_AIRPORTS)
    df = df[~still_used].copy()

    # Order matters: F3 corrects coordinates, and F1 judges a flight by the Haversine distance
    # between them, so the rejoin has to land first. F2 then only has to reach the two airports
    # the table does not list.
    f3_stats = rejoin_airport_metadata(df, airports)
    f2_stats = fix_airport_country_nulls(df, airports)
    f1_stats = fix_zero_duration_flights(df, load_route_durations(in_dir))
    # Its own stream: the main one feeds H4/A1 addresses and A5 prices, and consuming from it
    # here would shift every downstream draw and rewrite the whole KB.
    f4_stats = fix_zero_flight_numbers(df, np.random.default_rng(RNG_SEED + 2))

    df.to_csv(os.path.join(out_dir, 'flights.csv'), index=False)
    airports.to_csv(os.path.join(out_dir, 'VALID_airports_id.csv'), index=False)

    return {'flights': len(df), 'airports': len(airports),
            'airports_dropped': n_airports_in - len(airports),
            'barcelona_flights_dropped': int(still_used.sum()),
            'f3_rejoined': f3_stats,
            'f4_flight_numbers': f4_stats['fixed'],
            'f1_zero_fixed': f1_stats['fixed_zero'],
            'f1_neg_fixed':  f1_stats['fixed_negative'],
            'f1_total_fixed': f1_stats['fixed_total'],
            'f1_from_snapshot': f1_stats['from_snapshot'],
            'f1_from_estimate': f1_stats['from_estimate'],
            'f2_filled': f2_stats['filled'],
            'f2_unresolved': f2_stats['unresolved']}


def _query_dest_cities(row) -> list:
    """The cities a query's car could be rented in — its flight destinations."""
    try:
        r = ast.literal_eval(str(row['req_flight']))
        a = r.get('arrival_city')
        return [canonical_city(x) for x in (a if isinstance(a, list) else [a]) if x]
    except Exception:
        return []


def repoint_query_car_prices(q: pd.DataFrame, price_map: dict) -> dict:
    """Follow C2's reprice into the queries that quote a car price.

    A query does not just ask for a car, it names its price -- `req_car` carries
    {'price_per_day': '98.15', 'car_type': 'Minivan', ...} and the prose repeats it ("about 98.15
    USD per day"). Those numbers were read out of the KB at generation time, so repricing the cars
    without repricing the queries would leave 173 tasks quoting a price their destination no
    longer charges, and the agent would be asked to book a car that does not exist.

    Only quotes that resolve to a real car are moved. The 379 impossible tasks name fabricated
    types (Spaceship, Hovercar) at invented prices — that fabrication IS their ground truth, and
    rewriting it would make them solvable.
    """
    stats = {'req_car_updated': 0, 'text_updated': 0, 'unresolved': 0, 'skipped_fabricated': 0}
    for idx, row in q.iterrows():
        try:
            rc = ast.literal_eval(str(row['req_car']))
        except Exception:
            continue
        if not isinstance(rc, dict):
            continue
        old_raw, car_type = str(rc.get('price_per_day', '')), str(rc.get('car_type', ''))
        if not old_raw or not car_type:
            continue
        try:
            old = round(float(old_raw), 2)
        except ValueError:
            continue

        new = None
        for city in _query_dest_cities(row):
            if (city, car_type, old) in price_map:
                new = price_map[(city, car_type, old)]
                break
        if new is None:
            # A fabricated car (impossible task) or a quote no city matches. Leave it alone.
            stats['skipped_fabricated' if float(row.get('impossible', 0) or 0) == 1.0
                  else 'unresolved'] += 1
            continue

        rc['price_per_day'] = f'{new:.2f}'
        q.at[idx, 'req_car'] = str(rc)
        stats['req_car_updated'] += 1

        # The prose quotes the same number: "... about 98.15 USD per day."
        text = str(row['query'])
        new_text = re.sub(rf'\b{re.escape(old_raw)}\b(?=\s*USD per day)', f'{new:.2f}', text)
        if new_text != text:
            q.at[idx, 'query'] = new_text
            stats['text_updated'] += 1
    return stats


def process_queries(in_path: str, out_path: str, price_map: dict = None) -> dict:
    q = pd.read_csv(in_path)
    n_before = len(q)
    mask_drop = (
        q['query'].astype(str).str.contains('Barcelona', case=False, na=False) |
        q['req_flight'].fillna('').astype(str).str.contains('Barcelona', na=False)
    )
    dropped = q[mask_drop]
    kept = q[~mask_drop].copy()

    car = repoint_query_car_prices(kept, price_map) if price_map else {}

    kept.to_csv(out_path, index=False)
    return {'before': n_before, 'after': len(kept), 'dropped': int(mask_drop.sum()),
            'dropped_sample': dropped['query'].head(3).tolist(), 'car_prices': car}


# ----------------------------------------------------------------------------
# Main orchestration
# ----------------------------------------------------------------------------
def main():
    rng = np.random.default_rng(RNG_SEED)
    audit = {
        'hotel':       {'files_in': 0, 'files_out': 0, 'records_in': 0, 'records_out': 0,
                        'h1_total': 0, 'h2_total': 0, 'h4_total': 0,
                        'barcelona_skipped': []},
        'attraction':  {'files_in': 0, 'files_out': 0, 'records_in': 0, 'records_out': 0,
                        'a1_total': 0, 'a3_total': 0, 'barcelona_skipped': []},
        'car':         {'files_in': 0, 'files_out': 0, 'records_in': 0, 'records_out': 0,
                        'c2_repriced': 0, 'barcelona_skipped': []},
        'flight':      {},
        'query':       {},
        'locale_coverage': defaultdict(int),
        'fallback_countries': defaultdict(int),
        'coord': {},
    }

    # Wipe the output tree first. The build writes file-by-file and never used to clear, so a
    # renamed or dropped city left its old file behind to be served alongside the new one -- and
    # F5 renames 37 of them. v2 is fully regenerated from v1 in a few seconds, so there is
    # nothing here worth preserving.
    shutil.rmtree(OUT_V2, ignore_errors=True)

    # C2: cars get their own stream (main feeds H4/A1 addresses and A5 prices, seed+1 C1
    # coordinates, seed+2 F4 flight numbers), so repricing them leaves every other domain
    # bit-identical. car_price_map carries each (city, type, old price) -> new price forward to
    # the queries that quote it.
    car_mults = city_cost_multipliers()
    car_rng = np.random.default_rng(RNG_SEED + 3)
    car_price_map = {}

    # C3: hotel restaurant ratings, seed+4 — again its own stream so the reprice leaves every
    # other hotel column byte-identical.
    ror_rng = np.random.default_rng(RNG_SEED + 4)
    ap_rng = np.random.default_rng(RNG_SEED + 5)   # attraction ticket_price localisation
    ac_rng = np.random.default_rng(RNG_SEED + 6)   # attraction price cap resample
    cr_rng = np.random.default_rng(RNG_SEED + 7)   # coordinate regen for 26 broken cities

    # F5: v1 filename stem -> canonical city, filled as each domain is walked and reused to
    # rename that domain's .npy embeddings to the same key.
    rename_map = {}

    # ----- COORDINATE AUDIT (C1) -----
    # Runs first: hotels and attractions of the same city must share one transform.
    print("Auditing city coordinates against OurAirports reference points...")
    coord = audit_city_coords()
    tfs = coord['transforms']
    audit['coord'] = coord['stats']
    audit['coord']['repaired_cities'] = sorted(tfs)
    cs = coord['stats']
    print(f"  surveyed={cs['surveyed']} passing={cs['passing']} "
          f"displaced={cs['displaced']} dispersed={cs['dispersed']} "
          f"unmatched={len(cs['unmatched'])}")

    # ----- HOTEL -----
    print("Processing hotel data...")
    os.makedirs(OUT_HOTEL, exist_ok=True)
    for f in sorted(glob.glob(os.path.join(IN_HOTEL, '*.csv'))):
        audit['hotel']['files_in'] += 1
        fn = os.path.basename(f)
        if fn in BARCELONA_FILES:
            audit['hotel']['barcelona_skipped'].append(fn)
            audit['hotel']['records_in'] += sum(1 for _ in open(f)) - 1
            continue
        stem = fn[:-len('_hotel.csv')]
        city = city_of_csv(f)
        rename_map[stem] = city
        out = os.path.join(OUT_HOTEL, city + '_hotel.csv')
        sub = process_hotel_file(f, out, rng, tfs.get(stem), ror_rng)
        audit['hotel']['files_out'] += 1
        audit['hotel']['records_in'] += sub['records']
        audit['hotel']['records_out'] += sub['records']
        audit['hotel']['h1_total'] += sub['h1_fixed']
        audit['hotel']['h2_total'] += sub['h2_fixed']
        audit['hotel']['h4_total'] += sub['h4_regen']
        audit['hotel']['c3_ror'] = audit['hotel'].get('c3_ror', 0) + sub['c3_ror']
        audit['hotel']['n1_case'] = audit['hotel'].get('n1_case', 0) + sub['n1_case']
        audit['hotel']['ag_normalized'] = audit['hotel'].get('ag_normalized', 0) + sub.get('ag_normalized', 0)
        audit['hotel']['ar_amenities'] = audit['hotel'].get('ar_amenities', 0) + sub.get('ar_amenities', 0)
        audit['hotel']['nd_name'] = audit['hotel'].get('nd_name', 0) + sub.get('nd_name', 0)
        audit['hotel']['lr_lowrating'] = audit['hotel'].get('lr_lowrating', 0) + sub.get('lr_lowrating', 0)
        audit['hotel'].setdefault('ag_unmapped', set()).update(sub.get('ag_unmapped', set()))
        # Track locale coverage by reading country
        d = pd.read_csv(out, usecols=['country'])
        for c in d['country'].value_counts().items():
            cn, cnt = c
            if cn in LOCALE:
                audit['locale_coverage'][cn] += cnt
            else:
                audit['fallback_countries'][cn] += cnt
    print(f"  files in={audit['hotel']['files_in']} out={audit['hotel']['files_out']}")
    print(f"  records in={audit['hotel']['records_in']} out={audit['hotel']['records_out']}")
    print(f"  H1 fixed: {audit['hotel']['h1_total']}")
    print(f"  H2 fixed: {audit['hotel']['h2_total']}")
    print(f"  H4 regen: {audit['hotel']['h4_total']}")
    print(f"  C3 restaurant rating regenerated: {audit['hotel'].get('c3_ror', 0)}")
    print(f"  N1 case-collision names disambiguated: {audit['hotel'].get('n1_case', 0)}")
    print(f"  AR amenities normalized (drop Michelin, add Restaurant): {audit['hotel'].get('ar_amenities', 0)}")
    print(f"  ND doubled-word names folded: {audit['hotel'].get('nd_name', 0)}")
    print(f"  LR low-rating 'comfortable' phrasing neutralized: {audit['hotel'].get('lr_lowrating', 0)}")
    print(f"  AG amenities_group variants normalized: {audit['hotel'].get('ag_normalized', 0)}"
          + (f"  UNMAPPED={sorted(audit['hotel']['ag_unmapped'])}" if audit['hotel'].get('ag_unmapped') else ""))

    # ----- ATTRACTION -----
    print("\nProcessing attraction data...")
    os.makedirs(OUT_ATTR, exist_ok=True)
    for f in sorted(glob.glob(os.path.join(IN_ATTR, '*.csv'))):
        audit['attraction']['files_in'] += 1
        fn = os.path.basename(f)
        if fn in BARCELONA_FILES:
            audit['attraction']['barcelona_skipped'].append(fn)
            audit['attraction']['records_in'] += sum(1 for _ in open(f)) - 1
            continue
        stem = fn[:-len('_attraction.csv')]
        city = city_of_csv(f)
        rename_map[stem] = city
        out = os.path.join(OUT_ATTR, city + '_attraction.csv')
        _ac = city_of_csv(f)
        sub = process_attraction_file(f, out, rng, tfs.get(stem), car_mults.get(_ac), ap_rng)
        audit['attraction']['files_out'] += 1
        audit['attraction']['records_in'] += sub['records']
        audit['attraction']['records_out'] += sub['records']
        audit['attraction']['a1_total'] += sub['a1_regen']
        audit['attraction']['a3_total'] += sub['a3_fixed']
        audit['attraction']['a5_total'] = audit['attraction'].get('a5_total', 0) + sub.get('a5_price_fixed', 0)
        audit['attraction']['ao_dropped'] = audit['attraction'].get('ao_dropped', 0) + sub.get('ao_dropped', 0)
        audit['attraction']['ap_price'] = audit['attraction'].get('ap_price', 0) + sub.get('ap_price', 0)
        audit['attraction']['af_group'] = audit['attraction'].get('af_group', 0) + sub.get('af_group', 0)
        audit['attraction']['af_facilities'] = audit['attraction'].get('af_facilities', 0) + sub.get('af_facilities', 0)
        audit['attraction']['av_hours'] = audit['attraction'].get('av_hours', 0) + sub.get('av_hours', 0)
        audit['attraction']['am_name'] = audit['attraction'].get('am_name', 0) + sub.get('am_name', 0)
        audit['attraction']['nc_case'] = audit['attraction'].get('nc_case', 0) + sub.get('nc_case', 0)
        audit['attraction']['bp_persona'] = audit['attraction'].get('bp_persona', 0) + sub.get('bp_persona', 0)
        audit['attraction']['nw_hours'] = audit['attraction'].get('nw_hours', 0) + sub.get('nw_hours', 0)
    print(f"  files in={audit['attraction']['files_in']} out={audit['attraction']['files_out']}")
    print(f"  records in={audit['attraction']['records_in']} out={audit['attraction']['records_out']}")
    print(f"  A1 regen: {audit['attraction']['a1_total']}")
    print(f"  A3 fixed: {audit['attraction']['a3_total']}")
    print(f"  AO open_hours outliers dropped: {audit['attraction'].get('ao_dropped', 0)}")
    print(f"  A5 pegged-price fixed: {audit['attraction'].get('a5_total', 0)}")
    print(f"  AP ticket_price localised (Historic+Museum): {audit['attraction'].get('ap_price', 0)}")
    _ac = cap_attraction_prices(OUT_ATTR, ac_rng)
    audit['attraction']['ac_capped'] = _ac['ac_capped']
    print(f"  AC over-cap prices resampled: {_ac['ac_capped']}")
    print(f"  AF facilities_group variants folded: {audit['attraction'].get('af_group', 0)}")
    print(f"  AF facilities case-twins folded: {audit['attraction'].get('af_facilities', 0)}")
    print(f"  AV impossible-visit theme parks widened: {audit['attraction'].get('av_hours', 0)}")
    print(f"  AM mojibake names fixed: {audit['attraction'].get('am_name', 0)}")
    print(f"  NC case-collision names disambiguated: {audit['attraction'].get('nc_case', 0)}")
    print(f"  BP business-persona restored: {audit['attraction'].get('bp_persona', 0)}")
    print(f"  NW overnight open_hours fixed: {audit['attraction'].get('nw_hours', 0)}")
    _cr = regenerate_city_coords(cr_rng)
    audit['attraction']['cr'] = _cr
    print(f"  CR coords regenerated: {_cr['cr_rows']} rows in {_cr['cr_cities']} broken cities")

    # ----- CAR -----
    print("\nProcessing car data...")
    os.makedirs(OUT_CAR, exist_ok=True)
    for f in sorted(glob.glob(os.path.join(IN_CAR, '*.csv'))):
        audit['car']['files_in'] += 1
        fn = os.path.basename(f)
        if fn in BARCELONA_FILES:
            audit['car']['barcelona_skipped'].append(fn)
            audit['car']['records_in'] += sum(1 for _ in open(f)) - 1
            continue
        stem = fn[:-len('_rental_cars.csv')]
        city = city_of_csv(f)
        rename_map[stem] = city
        out = os.path.join(OUT_CAR, city + '_rental_cars.csv')
        sub = process_car_file(f, out, car_mults.get(city), car_rng, car_price_map)
        audit['car']['c2_repriced'] += sub.get('c2_repriced', 0)
        audit['car']['files_out'] += 1
        audit['car']['records_in'] += sub['records']
        audit['car']['records_out'] += sub['records']
    print(f"  files in={audit['car']['files_in']} out={audit['car']['files_out']}")
    print(f"  C2 repriced by city cost level: {audit['car']['c2_repriced']}")

    # Group embeddings (amenities/facilities/extra_services) are NOT produced here: they are Titan
    # v2 vectors generated by generate_embeddings.py against these CSVs. The v1 Qwen .npy files are
    # a different model and are deliberately not carried over. Run generate_embeddings.py after this.

    # ----- FLIGHT -----
    print("\nProcessing flight data...")
    audit['flight'] = process_flights(IN_FLIGHT, OUT_FLIGHT)
    print(f"  flights: {audit['flight']['flights']}, airports: {audit['flight']['airports']}")
    print(f"  F1 too-fast-to-fly fixed: {audit['flight']['f1_zero_fixed']}")
    print(f"  F1 negative-duration fixed: {audit['flight']['f1_neg_fixed']}")
    print(f"  F1 total fixed: {audit['flight']['f1_total_fixed']} "
          f"(judged at the snapshot date's offset, 2025-08-01)")

    # ----- QUERY -----
    print("\nProcessing queries...")
    audit['query'] = process_queries(IN_QUERY, OUT_QUERY, car_price_map)
    print(f"  queries before={audit['query']['before']} after={audit['query']['after']} "
          f"dropped={audit['query']['dropped']}")

    # ----- AUDIT REPORT -----
    write_audit(audit)
    print(f"\nDone. Audit report: {AUDIT_PATH}")


def write_audit(a):
    lines = []
    lines.append("# TravelBench KB v2 — Audit Report\n")
    lines.append(f"Generated by `build_v2_kb.py` (seed={RNG_SEED}).\n")
    lines.append("\n## 1. Hotel data\n")
    lines.append(f"- Files in:  {a['hotel']['files_in']}")
    lines.append(f"- Files out: {a['hotel']['files_out']} (skipped Barcelona: {a['hotel']['barcelona_skipped']})")
    lines.append(f"- Records in:  {a['hotel']['records_in']}")
    lines.append(f"- Records out: {a['hotel']['records_out']}")
    lines.append(f"- C3 restaurant rating regenerated from star (5-star anchored at mean 4.3): "
                 f"{a['hotel'].get('c3_ror', 0)}")
    lines.append(f"- N1 case-collision hotel names disambiguated: {a['hotel'].get('n1_case', 0)}")
    lines.append(f"- H1 rating denominator fixed: {a['hotel']['h1_total']}")
    lines.append(f"- H2 half-star truncation fixed: {a['hotel']['h2_total']}")
    lines.append(f"- H4 address regenerated: {a['hotel']['h4_total']}")

    lines.append("\n## 2. Attraction data\n")
    lines.append(f"- Files in:  {a['attraction']['files_in']}")
    lines.append(f"- Files out: {a['attraction']['files_out']} (skipped Barcelona: {a['attraction']['barcelona_skipped']})")
    lines.append(f"- Records in:  {a['attraction']['records_in']}")
    lines.append(f"- Records out: {a['attraction']['records_out']}")
    lines.append(f"- A1 address regenerated: {a['attraction']['a1_total']}")
    lines.append(f"- A3 open-hours overview fixed: {a['attraction']['a3_total']}")
    lines.append(f"- A5 pegged outlier ticket_price fixed: {a['attraction'].get('a5_total', 0)} "
                 f"(was $1,654.66 / $2,059.20 hard-coded; replaced with same-type median +/- 20% noise)")

    lines.append("\n## 3. Car data\n")
    lines.append(f"- Files in:  {a['car']['files_in']}")
    lines.append(f"- Files out: {a['car']['files_out']} (skipped Barcelona: {a['car']['barcelona_skipped']})")
    lines.append(f"- Records in:  {a['car']['records_in']}")
    lines.append(f"- Records out: {a['car']['records_out']}")

    lines.append("\n## 4. Flight data\n")
    lines.append(f"- flights.csv rows: {a['flight']['flights']} (was 109,262)")
    lines.append(f"- VALID_airports_id.csv rows: {a['flight']['airports']}")
    lines.append(f"- F1 too fast to fly the distance, rewritten: {a['flight'].get('f1_zero_fixed', 0)}")
    lines.append(f"- F1 negative duration (arrival before departure), rewritten: {a['flight'].get('f1_neg_fixed', 0)}")
    lines.append(f"- F1 total rewritten: {a['flight'].get('f1_total_fixed', 0)} of 107,195 "
                 f"({a['flight'].get('f1_total_fixed', 0) / 107195 * 100:.2f}%)")
    lines.append("- Timezones resolved per airport from IANA tzdata (see `AIRPORT_TZ`), read at the")
    lines.append("  offset in force on the snapshot date 2025-08-01 -- the snapshot is dated, so no")
    lines.append("  DST inference is involved. Verified against IANA for all 392 airports.")
    lines.append(f"- Repaired durations taken from the source snapshot's real median for the same")
    lines.append(f"  route: {a['flight'].get('f1_from_snapshot', 0)}; "
                 f"route absent from the snapshot, Haversine estimate used: "
                 f"{a['flight'].get('f1_from_estimate', 0)}")
    lines.append(f"- F2 airport country/continent backfilled (failed ISO-2 vs full-name join): "
                 f"{a['flight'].get('f2_filled', 0)}"
                 + (f"; unresolved: {a['flight']['f2_unresolved']}"
                    if a['flight'].get('f2_unresolved') else ""))
    lines.append(f"- Barcelona airports dropped from VALID_airports_id.csv: "
                 f"{a['flight'].get('airports_dropped', 0)} (BCN, BLA); "
                 f"Barcelona flights still present and removed: "
                 f"{a['flight'].get('barcelona_flights_dropped', 0)}")
    f3 = a['flight'].get('f3_rejoined') or {}
    if f3:
        lines.append("- F3 airport metadata re-derived from VALID_airports_id.csv by IATA "
                     "(cells changed):")
        for k in sorted(f3):
            if f3[k]:
                lines.append(f"    - {k}: {f3[k]}")
        lines.append("  Fixes NRT's split city label (Tokyo/Narita), TAO's decommissioned name")
        lines.append("  (Liuting, closed 2021), and six airports' coordinates incl. PKX (1.4 km).")
        lines.append("  Continent quoting stripped: the release stored `\"NA\"` with literal quotes")
        lines.append("  because bare NA reads as null under pandas defaults.")
    lines.append(f"- F4 flight numbers redrawn from `XX 0000` (no airline has a flight 0): "
                 f"{a['flight'].get('f4_flight_numbers', 0)}")

    lines.append("\n## 5. Queries\n")
    lines.append(f"- before: {a['query']['before']}, after: {a['query']['after']}, dropped: {a['query']['dropped']}")
    for s in a['query']['dropped_sample']:
        lines.append(f"    e.g. \"{s[:120]}...\"")

    c = a.get('coord') or {}
    if c:
        lines.append("\n## 6. Coordinate audit (Bug C1)\n")
        lines.append("Every city's pooled hotel+attraction coordinate cloud was checked against its")
        lines.append("OurAirports reference point, matched on (city, country) and taking the nearest")
        lines.append("airport where a city has several.\n")
        lines.append(f"- Cities surveyed: {c.get('surveyed', 0)}")
        lines.append(f"- **Passed, left untouched: {c.get('passing', 0)}**")
        lines.append(f"- Failed, re-anchored: {c.get('displaced', 0)} "
                     f"(cloud centre >{DISPLACED_KM:.0f} km from the airport)")
        lines.append(f"- Failed, rescaled: {c.get('dispersed', 0)} "
                     f"(90th-pct radius >{DISPERSED_KM:.0f} km)")
        lines.append(f"- No airport entry for (city, country): {len(c.get('unmatched', []))} "
                     f"{c.get('unmatched', [])}")
        lines.append(f"- Passing cities' median 90th-pct radius: {c.get('donor_median_radius', 0):.1f} km "
                     f"(repair targets are drawn from this same distribution)")
        if c.get('repaired_cities'):
            lines.append(f"\nRepaired: {', '.join(c['repaired_cities'])}")

    lines.append("\n## 7. Locale coverage (Bug H4 / A1)\n")
    lines.append("Countries handled by locale-aware templates:\n")
    for c, n in sorted(a['locale_coverage'].items(), key=lambda x: -x[1])[:30]:
        lines.append(f"- {c}: {n} hotel records")
    if a['fallback_countries']:
        lines.append("\nCountries falling back to generic English template:")
        for c, n in sorted(a['fallback_countries'].items(), key=lambda x: -x[1]):
            lines.append(f"- {c}: {n} hotel records")
    else:
        lines.append("\nNo countries fell back — all countries covered by LOCALE table.")

    lines.append("\n## 7. Final totals\n")
    total_records = (a['hotel']['records_out'] + a['attraction']['records_out']
                     + a['car']['records_out'] + a['flight']['flights'])
    lines.append(f"- Total records (hotel + attraction + car + flight): **{total_records}** (was 215,164)")
    lines.append(f"- Total queries: **{a['query']['after']}** (was 1,500)")
    lines.append(f"- Total cities: **376** (was 377, Barcelona removed)")

    lines.append("\n## 8. Paper number updates required\n")
    lines.append("- Abstract: '1,500 queries' → '1,493 queries'")
    lines.append("- Abstract: 'over 215K records' → 'over 212K records' (or precise)")
    lines.append("- Abstract / Intro: '377 cities' → '376 cities'")
    lines.append("- §3.2 Itinerary Complexity percentages may shift slightly")
    lines.append("- §4.1 Constraint counts may shift slightly")
    lines.append("- All experimental tables and headline numbers: re-run + update")

    with open(AUDIT_PATH, 'w') as fh:
        fh.write('\n'.join(lines))


if __name__ == '__main__':
    main()
