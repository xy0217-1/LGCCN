class Labelexpand():
    def __init__(self,dataset):
        self.dataset=dataset


    def getlabeldict(self):
        if self.dataset=="HuffPost":
            return self._get_huffpost_classes()
        elif self.dataset=="20News":
            return self._get_20newsgroup_classes()
        elif self.dataset=="Amazon":
            return self._get_amazon_classes()
        elif self.dataset=="Reuters":
            return self._get_reuters_classes()
        else:
            print("Dataset ERROR")
        return
    def _get_20newsgroup_classes(self):
        label_dict = {
            0:'talk politics mideast',
            1:'science space',
            2:'misc forsale',
            3:'talk politics misc',
            4:'computer graphics',
            5:'science  encryption  encrypt secret',
            6:'computer windows x',
            7:'computer os ms windows misc',
            8:'talk politics guns',
            9:'talk religion misc',
            10:'rec autos',
            11:'science med chemistry medical science medicine',
            12:'computer sys mac hardware',
            13:'science electronics',
            14:'rec sport hockey',
            15:'alt atheism',
            16:'rec motorcycles',
            17:'computer system ibm pc hardware',
            18:'rec sport baseball',
            19:'soc religion christian',
        }

        return  label_dict


    def _get_amazon_classes(self):

        label_dict = {
            0:'Amazon Instant Video is a subscription video on-demand over-the-top streaming and rental service of Amazon.com',
            1:'Apps for Android is a computer program or software application designed to run on a Android device , like game app , music app , browser app',
            2:'Automotive is concerned with self-propelled vehicles or machines',
            3:'Baby means Baby products that moms will use for their kids, like baby tracker , baby bottles , bottle warmer , baby nipple',
            4:'Beauty products like Cosmetics are constituted from a mixture of chemical compounds derived from either natural sources or synthetically created ones',
            5:'Books are long written or printed literary compositions , which tell us good stories or philosophy',
            6:'CDs and DVDs are digital optical disc data storage formats to store and play digital audio recordings or music , something similar including compact disc (CD), vinyl, audio tape, or another medium',
            7:'Cell Phones and Accessories refer to mobile phone and some hardware designed for the phone like microphone , headset',
            8:'Clothing ，Shoes and Jewelry are items worn on the body to protect and comfort the human or for personal adornment',
            9:'Albums and Digital Music are collections of audio recordings including popular songs and splendid music',
            10:'Electronics refer to electronic devices, or the part of a piece of equipment that consists of electronic devices',
            11:'Grocery and Gourmet Food refer to stores primarily engaged in retailing a general range of food products',
            12:'Health and Personal Care refer to consumer products used in personal hygiene and for beautification',
            13:'Home and Kitchen refer to something used in Home and Kitchen such as Kitchenware , Tableware , cleaning tools',
            14:'Kindle Store is an online e-book e-commerce store operated by Amazon as part of its retail website and can be accessed from any Amazon Kindle',
            15:'Movies and TV is a work of visual art that tells a story and that people watch on a screen or television or a showing of a motion picture especially in a theater',
            16:'Musical Instruments are devices created or adapted to make musical sounds',
            17:'Office Products are consumables and equipment regularly used in offices by businesses and other organizations',
            18:'Patio Lawn and Garden refer to some tools and devices used in garden or lawn',
            19:'Pet Supplies refer to food or other consumables or tools that will be used when you keep a pet , like dog food , cat treat , pet toy',
            20:'Sports and Outdoors refer to some tools and sport equipment used in outdoor sports',
            21:'Tools and Home Improvement refer to hand tools or implements used in the process of renovating a home',
            22:'Toys and Games are something used in play , usually undertaken for entertainment or fun, and sometimes used as an educational tool.',
            23:'Video Games or Computer games are electronic games that involves interaction with a user interface or input device to generate visual feedback , which include arcade games , console games , and personal computer (PC) games',
        }

        return  label_dict


    def _get_huffpost_classes(self):
        label_dict = {
            0:'politics government court amendment congressional debate runoff democrat republican Political figures',
            1:'wellness healthy sport body exercise therapist workout training sleep yoga happiness diet',
            2:'entertainment enjoy Movie Film and television works and Entertainment star and Entertainment news',
            3:'travel travelers trip Tourism Tourist destination flight flights airport airlines vacation italian hotel hotels map italy disney',
            4:'style and beauty beautiful photos adidas fashion clothes dress magazine covers photos makeup looks star',
            5:'parenting Parents parent mother mom dad uncles kids daughter babies child baby teens',
            6:'healthy living health care drug medical medicare medicaid disease virus deaths',
            7:'queer voices LGBT LGBTQ lesbian gay hiv homosexual love straight people trans , nonbinary community gendering transgender Sexual orientation coming out pride ',
            8:'food foods and drink fruit recipes recipe delicious sandwich pizza chicken wine',
            9:'business company uber amazon wells fargo bank bankrupt billionaire stock leader ceo lead worker',
            10:'comedy Interesting thing Gossip jokes hilariously funny jimmy show stephen',
            11:'sports olympic olympics game team athletes player players bowl winter hockey baseball basketball football soccer gymnastics skate',
            12:'black voices racist racism police cop white people rapper black men martin luther king',
            13:'home & living apartment home butler home design furniture bedroom holiday christmas ',
            14:'parents parenting parent mother mom: dad uncles kids child baby babies teens family',
            15:'the world post korea u.s. china france war attack nuclear missile iran refugees isis egypt syria america referendum ',
            16:'weddings wedding engagement honeymoon bridal marriage love couples dresses brides bride groom bridesmaid romance newlyweds ring knot guests',
            17:'women woman female harassment men sexual gender sexist feminism abortions abortion',
            18:'impact refugees fight hurricane homelessness shelters donate poverty hunger homeless',
            19:'divorce divorced uncoupling breakup single dating spouse husband wife children marriage married family  infidelity alimony',
            20:'crime shooting shooter shot hostage crime murder gunman suspect charged arrested arrest inmate police victim',
            21:'media advertisers media twitter editor journalists journalism news editor journalist newspaper',
            22:'weird news weird halloween fark quiz',
            23:'green climate hurricane storm environment environmental climate change wildfire coal animal whale gorilla',
            24:'world post iran world isis war greece china syrian russia africa crisis america yemen mediterranean poland elections migrants philippines',
            25:'religion pope muslim meditation ramadan faith church muslims christian religious christians god religion christianity evangelicals catholic jesus',
            26:'style clothes beauty fashion hair dress makeup prince clothing carpet',
            27:'science scientists space nasa science earth brain telescope galaxy astronomers',
            28:'world news reuters president embassy attack australia israel zimbabwe myanmar jerusalem',
            29:'taste food foods sweets meal foods barbecue salads wine coffee recipes cooking',
            30:'tech facebook apple google iphone twitter uber microsoft instagram samsung users app encryption android hackers cyber web hackers',
            31:'money credit tax financial finances lottery investor savings costs buy debt mortgage banks bank money',
            32:'arts art artist stage ballet music photography photographer nighter theatre dance',
            33:'fifty midlife retire age care grandma mother retirement aging alzheimer older grandkids childhood',
            34:'good news selfless kittens dog cat rescue rescued adorable',
            35:'arts and culture book new artist women museum books broadway history authors sculpture potter',
            36:'environment week climate animal tigers giraffes animals weather tornado  oil species chemicals',
            37:'college professors faculty chancellor professor freshman student campus university universities colleges fraternity commencement',
            38:'latino voices latinos latina immigrants immigration spanish latin mexico mexican hispanic',
            39:'culture and arts photos image blog artist artists culture gallery exhibition photo theatre photography paintings',
            40:'education classrooms classroom education learning student students school schools school districts stem educational college teacher teachers teaching'
        }
        return  label_dict

    def _get_reuters_classes(self):
        label_dict = {
            0:'acquisition merge If a company or business person makes an acquisition , they buy another company or part of a company',
            1:'aluminium Aluminium is a lightweight metal used, for example, for making cooking equipment and aircraft parts',
            2:'trade deficit , current account deficit mean financial situation in the red , shortage , decrease , decline' ,
            3:'cocoa Cocoa is a brown powder made from the seeds of a tropical tree. It is used in making chocolate . It is usually as goods for trade' ,
            4:'coffee Coffee is the roasted beans or powder from which the drink is made',
            5:'copper Copper is reddish brown metal that is used to make things such as coins and electrical wires',
            6:'cotton Cotton is a plant and which produces soft fibres used in making cotton cloth',
            7:'inflation Inflation is a general increase in the prices of goods and services in a country',
            8:'oil Oil is a smooth , thick liquid that is used as a fuel and for making the parts of machines move smoothly. Oil is found underground',
            9:'profit A profit is an amount of money that you gain when you are paid more for something than itcost you to make, get, or do it',
            10:'gdp gnp gross domestic product gross national product In economics, a country GDP is the total value of goods and services produced within a country in a year',
            11:'gold Gold is a valuable , yellow-coloured metal that is used for making jewellery and ornaments, and as an international currency',
            12:'grain Grain is a cereal , especially wheat or corn , that has been harvested and is used for food or in trade',
            13:'rate A rate is the amount of money that is charged for goods or services or ',
            14:'industrial production Industrial production refers to the production output and the output of industrial establishments and covers sectors such as mining, manufacturing, electricity, gas and steam and air-conditioning',
            15:'steel Steel is a very strong metal which is made mainly from iron. Steel is used for making many things, for example bridges, buildings, vehicles, and cutlery . It is a important merchandise',
            16:'unemployment Unemployment is the fact that people who lose jobs, want jobs cannot get them',
            17:'cattle Cattle are cows and bulls , including animals living in farm',
            18:'treasury bank the Treasury is the government department that deals with the country finances',
            19:'money supply Related policies and news on money supply from the central financial departm',
            20:'gas Gas is a substance like air and burns easily. It is used as a fuel for cooking and heat',
            21:'orange a round sweet fruit that has a thick orange skin and an orange centre , which is goods in tr',
            22:'reserves foreign reserves , gold and currency reserves , The amount of foreign currency stored by the central b',
            23:'retail Retail is the activity of selling products direct to the public, usually in small quantit',
            24:'rubber  Rubber is a strong, waterproof, elastic substance made from the juice of a tropical tree or produced chemically. It is used for making tyres, boots, and other produ',
            25:'ship a large boat for travelling on water, especially across the sea',
            26:'sugar Sugar is a sweet substance that is used to make food and drinks sweet. It is usually as goods for trade',
            27:'tin A tin is a metal container which is filled with food and sealed in order to preserve the food for long periods of time',
            28:'tariffs A tariff is a tax that a government collects on goods coming into a country',
            29:'oils and fats tax mean the tax in oils and fats which Promulgated by the European Union',
            30:'producer price wholesale A producer price index (PPI) is a price index that measures the average changes in prices received by domestic producers for their output.',
        }

        return  label_dict


