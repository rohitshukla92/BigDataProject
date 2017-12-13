#!/usr/bin/env python
# -*- coding: utf-8 -*- 
from __future__ import absolute_import, print_function
from nltk.stem.snowball import SnowballStemmer
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import json 
import os
import time
import sys
from HTMLParser import HTMLParser
from itertools import chain
import nltk
nltk.download('punkt')
from nltk import word_tokenize,sent_tokenize
import datetime
import csv
import codecs


# Go to http://apps.twitter.com and create an app.
# The consumer key and secret will be generated for you after
consumer_key="Pr5qbY4DjgNq7sbDgCo8sK3fu"
consumer_secret="VSJASScDTyKzddAQue2TdDB98dn4qCX22IgVTiuMBp1wPuPXeZ"

# After the step above, you will be redirected to your app's page.
# Create an access token under the the "Your access token" section
access_token="930208627708882944-EtqVQ8YYXINx9Z7mZoGvVkKvb6Rrxl4"
access_token_secret="IoP6Wb1628namo6s3AZWgsMtzpCgWpJjHmrj5zNSkLodj"
stemmer=SnowballStemmer("english")

words=["hunger","craving","desire","famine","greed","longing","lust","starvation","yearning","ache","appetence",
"emptiness","esurience","famishment","gluttony","greediness","hungriness","mania","ravenous","vacancy","void","voracity",
"want","yen","Bankruptcy","debt","deficit","difficulty","hardship","lack","scarcity","shortage","underdevelopment","abjection",
"aridity","barrenness","beggary","dearth","deficiency","depletion","distress","emptiness","exiguity","impecuniousness",
"impoverishment","inadequacy","Indigence","insolvency","insufficiency","meagerness","necessity","pass","paucity",
"pauperism","Pennilessness","penury","pinch","poorness","privation","reduction","straits","vacancy","necessitousness","Drought","misery",
"destitution","paucity","malnourished","Bulimic","emaciated","thin","sickly","looking","without","appetite","thin",
"Poor","Destitute","impoverished","indigent","low","meager","needy","penniless","poverty-stricken",
"Underprivileged","bankrupt","down-and-out","flat","insolvent","scanty","suffering","bad","off","Beggared","beggarly","dirt","poor","empty-handed",
"flat","broke","fortuneless","impecunious","Moneyless","necessitous","pauperized","penurious","pinched","reduced","stone","strapped","unprosperous"]

countries=[["Afghanistan"], ["Albania"], ["Algeria"], ["American Samoa"],
["Andorra"], ["Angola"], ["Anguilla"], ["Antarctica"], ["Antigua and Barbuda"],
["Argentina"], ["Armenia"], ["Aruba"], ["Australia","Australian Capital Territory", "New South Wales", "Northern Territory", "Queensland", "South Australia", "Tasmania", "Victoria", "Western Australia"],
["Austria"], ["Azerbaijan"], ["Bahamas"], ["Bahrain"], ["Bangladesh"], ["Barbados"], 
["Belarus"], ["Belgium"], ["Belize"], ["Benin"], ["Bermuda"], ["Bhutan"], ["Bolivia"], 
["Bosnia and Herzegovina"],["Botswana"], ["Bouvet Island"],["Brazil","Acre", "Alagoas", "Amapa", "Amazonas", "Bahia", "Ceara", "Distrito Federal", "Espirito Santo", "Goias", "Maranhao", "Mato Grasso", "Mato Grosso do Sul", "Minas Gerais", "Parana", "Paraiba", "Pará", "Pernambuco", "Piaua", "Rio Grande do Norte", "Rio Grande do Sul", "Rio de Janeiro", "Rondônia", "Roraima", "Santa Catarina", "Sergipe", "Sao Paulo", "Tocantins"],
["British Indian Ocean Territory"], ["Brunei Darussalam"], ["Bulgaria"], ["Burkina Faso"], ["Burundi"],
["Cambodia"], ["Cameroon"], ["Canada","Alberta", "British Columbia", "Manitoba", "New Brunswick", "Newfoundland", "Northwest Territories", "Nova Scotia", "Nunavut", "Ontario", "Prince Edward Island", "Quebec", "Saskatchewan", "Yukon"], 
["Cape Verde"], ["Cayman Islands"], ["Central African Republic"], ["Chad"], ["Chile"], 
["China"], ["Christmas Island"], ["Cocos (Keeling Islands)"], ["Colombia"], ["Comoros"], ["Congo"], 
["Cook Islands"], ["Costa Rica"], ["Cote D'Ivoire (Ivory Coast)"], ["Croatia (Hrvatska"], ["Cuba"], 
["Cyprus"], ["Czech Republic"], ["Denmark"], ["Djibouti"], ["Dominica"], ["Dominican Republic"], ["East Timor"],
["Ecuador"], ["Egypt"], ["El Salvador"], ["Equatorial Guinea"], ["Eritrea"], ["Estonia"], ["Ethiopia"], 
["Falkland Islands (Malvinas)"], ["Faroe Islands"], ["Fiji"], ["Finland"], ["France"], ["France, Metropolitan"],
["French Guiana"], ["French Polynesia"], ["French Southern Territories"], ["Gabon"], ["Gambia"], ["Georgia"], 
["Germany","Baden-Württemberg", "Bayern", "Berlin", "Brandenburg", "Bremen", "Hamburg", "Hessen", "Mecklenburg-Vorpommern", "Niedersachsen", "Nordrhein-Westfalen", "Rheinland-Pfalz", "Saarland", "Sachsen", "Sachsen-Anhalt", "Schleswig-Holstein", "Thüringen"], 
["Ghana"], ["Gibraltar"], ["Greece"], ["Greenland"], ["Grenada"], ["Guadeloupe"], ["Guam"], ["Guatemala"], 
["Guinea"], ["Guinea-Bissau"], ["Guyana"], ["Haiti"], ["Heard and McDonald Islands"], ["Honduras"], 
["Hong Kong"], ["Hungary"], 
["Iceland"], ["India","Andhra Pradesh","Arunachal Pradesh","Assam","Bihar","Chhattisgarh","Goa","Gujarat","Haryana","Himachal Pradesh","Jammu and Kashmir","Jharkhand","Karnataka","Kerala","Madhya Pradesh","Maharashtra","Manipur","Meghalaya","Mizoram","Nagaland","Odisha","Punjab","Rajasthan","Sikkim","Tamil Nadu","Telangana","Tripura","Uttar Pradesh","Uttarakhand","West Bengal","Andaman and Nicobar","Chandigarh","Dadra and Nagar Haveli","Daman and Diu","Lakshadweep","Delhi","Puducherry"], 
["Indonesia"], ["Iran"], ["Iraq"], ["Ireland","Carlow", "Cavan", "Clare", "Cork", "Donegal", "Dublin", "Galway", "Kerry", "Kildare", "Kilkenny", "Laois", "Leitrim", "Limerick", "Longford", "Louth", "Mayo", "Meath", "Monaghan", "Offaly", "Roscommon", "Sligo", "Tipperary", "Waterford", "Westmeath", "Wexford", "Wicklow"],
["Israel"], ["Italy"], ["Jamaica"], ["Japan"], ["Jordan"], ["Kazakhstan"], ["Kenya"], ["Kiribati"], ["Korea (North)"], ["Korea (South)"], ["Kuwait"], ["Kyrgyzstan"], ["Laos"], 
["Latvia"], ["Lebanon"], ["Lesotho"], ["Liberia"], ["Libya"], ["Liechtenstein"], ["Lithuania"], ["Luxembourg"], ["Macau"], ["Macedonia"], ["Madagascar"], ["Malawi"], ["Malaysia"], 
["Maldives"], ["Mali"], ["Malta"], ["Marshall Islands"], ["Martinique"], ["Mauritania"], ["Mauritius"], ["Mayotte"], ["Mexico","Aguascalientes", "Baja California Sur", "Baja California", "Campeche", "Chiapas", "Chihuahua", "Coahuila", "Colima", "Distrito Federal", "Durango", "Guanajuato", "Guerrero", "Hidalgo", "Jalisco", "Michoacán", "Morelos", "México", "Nayarit", "Nuevo León", "Oaxaca", "Puebla", "Querétaro", "Quintana Roo", "San Luis Potosi", "Sinaloa", "Sonora", "Tabasco", "Tamaulipas", "Tlaxcala", "Veracruz", "Yucatán", "Zacatecas"],
["Micronesia"], ["Moldova"], ["Monaco"], ["Mongolia"], ["Montserrat"], ["Morocco"], ["Mozambique"], ["Myanmar"], ["Namibia"], ["Nauru"], ["Nepal"], ["Netherlands Antilles"], ["Netherlands","Drenthe", "Flevoland", "Friesland", "Gelderland", "Groningen", "Limburg", "Noord-Brabant", "Noord-Holland", "Overijssel", "Utrecht", "Zeeland", "Zuid-Holland"],
["New Caledonia"], ["New Zealand"], ["Nicaragua"], ["Niger"], ["Nigeria"], ["Niue"], ["Norfolk Island"], ["Northern Mariana Islands"], ["Norway"], ["Oman"], ["Pakistan"], ["Palau"], ["Panama"], ["Papua New Guinea"], ["Paraguay"], ["Peru"], ["Philippines"], ["Pitcairn"], ["Poland"], ["Portugal"], ["Puerto Rico"], ["Qatar"], ["Reunion"], ["Romania"], ["Russian Federation"], ["Rwanda"], ["S. Georgia and S. Sandwich Isls."], 
["Saint Kitts and Nevis"], ["Saint Lucia"], ["Saint Vincent and The Grenadines"], ["Samoa"], ["San Marino"], ["Sao Tome and Principe"], ["Saudi Arabia"], ["Senegal"], ["Seychelles"], ["Sierra Leone"], ["Singapore"], ["Slovak Republic"], ["Slovenia"], ["Solomon Islands"], ["Somalia"], ["South Africa"], ["Spain"], ["Sri Lanka"], ["St. Helena"], ["St. Pierre and Miquelon"], ["Sudan"], ["Suriname"], ["Svalbard and Jan Mayen Islands"], ["Swaziland"], 
["Sweden"], ["Switzerland"], ["Syria"], ["Taiwan"], ["Tajikistan"], ["Tanzania"], ["Thailand"], ["Togo"], ["Tokelau"], ["Tonga"], ["Trinidad and Tobago"], ["Tunisia"], ["Turkey"], ["Turkmenistan"], ["Turks and Caicos Islands"], ["Tuvalu"], ["US Minor Outlying Islands"], ["Uganda"], ["Ukraine"], ["United Arab Emirates"], ["United Kingdom"], 
["United States","Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware", "District Of Columbia", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming"], ["Uruguay"], ["Uzbekistan"], ["Vanuatu"], ["Vatican City State (Holy See)"], ["Venezuela"], ["Viet Nam"], ["Virgin Islands (British)"], ["Virgin Islands (US)"], ["Wallis and Futuna Islands"],
["Western Sahara"], ["Yemen"], ["Yugoslavia"], ["Zaire"], ["Zambia"], ["Zimbabwe"] ]

root_words=[]
for w in words:
    root_words.append(stemmer.stem(w))

class StdOutListener(StreamListener):
    """ A listener handles tweets that are received from the stream.
    This is a basic listener that just prints received tweets to stdout.
    """
    def on_data(self, data):
        root_tokens=[]
        try:
            data = json.loads(HTMLParser().unescape(data))
            #time_tweet = data['timestamp_ms']
            #date = datetime.datetime.fromtimestamp(int(time_tweet) / 1000)
            #new_date = str(date).split(" ") [0]
            tweet = data['text']
            date=data['created_at']
            tokens=nltk.word_tokenize(tweet)
            for t in tokens:
                root_tokens.append(stemmer.stem(t))            
            for rt in root_tokens:
                for p in tokens:
                    if rt in root_words and p in [j for i in countries for j in i]:
                        location=data['user']['location']
                        print(date,p,rt,location,tweet)     
            return True
        except BaseException, e:
            print('failed ondata',str(e))
            time.sleep(5)  

    def on_error(self, status):
        print(status)

if __name__ == '__main__':
    global words
    l = StdOutListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    stream = Stream(auth, l)
    stream.filter(track=["hunger","malnourishment","poverty","poor","famine","underdevelopment"])




