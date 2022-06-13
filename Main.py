import streamlit as st
import requests
from bs4 import BeautifulSoup
import json
from math import sqrt
from shapely.geometry import Point, Polygon
import pandas as pd
import re
import plotly_express as px
import numpy as np
import folium
from streamlit_folium import folium_static
import geopy.distance
from folium.plugins import HeatMap
import plotly.graph_objects as go
from io import StringIO, BytesIO
import base64
import matplotlib.pyplot as plt
from PIL import Image
with st.echo(code_location='below'):
    ####DOWNLOAD EXCEL AND PLOT FUNCTIONS
    ### Adapted FROM: (https://discuss.streamlit.io/t/how-to-add-a-download-excel-csv-function-to-a-button/4474/5)
    def generate_excel_download_link(df):
        towrite = BytesIO()
        df.to_excel(towrite, encoding="utf-8", index=False, header=True)  # write to BytesIO buffer
        towrite.seek(0)  # reset pointer
        b64 = base64.b64encode(towrite.read()).decode()
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="data_download.xlsx">Download Excel File of the graph</a>'
        return st.markdown(href, unsafe_allow_html=True)

    ### END FROM
    ### Adapted FROM: (https://discuss.streamlit.io/t/download-plotly-plot-as-html/4426/2)
    def generate_html_download_link(fig):
        towrite = StringIO()
        fig.write_html(towrite, include_plotlyjs="cdn")
        towrite = BytesIO(towrite.getvalue().encode())
        b64 = base64.b64encode(towrite.read()).decode()
        href = f'<a href="data:text/html;charset=utf-8;base64, {b64}" download="plot.html">Download Plot of the graph</a>'
        return st.markdown(href, unsafe_allow_html=True)

    ### END FROM
    ####END DOWNLOAD
    def embed_map(m):
        from IPython.display import HTML

        m.save('index.html')
        with open('index.html') as f:
            html = f.read()

        iframe = '<iframe srcdoc="{srcdoc}" style="width: 100%; height: 750px; border: none"></iframe>'
        srcdoc = html.replace('"', '&quot;')
        return HTML(iframe.format(srcdoc=srcdoc))
    def dist(x1, y1, x2, y2):
        return(sqrt((x1-x2)**2-(y1-y2)**2))
    def read_file(data_url):
        return pd.read_csv(data_url)
    #Парсинг сайта Википедии в поиске координат достопримечательностей Москвы, файл parsed_data.csv получен с его помощью
    #можете повторно запустить для проверки
    def parsing_mem():
        url = "https://ru.wikipedia.org/wiki/%D0%9A%D0%B0%D1%82%D0%B5%D0%B3%D0%BE%D1%80%D0%B8%D1%8F:%D0%9F%D0%B0%D0%BC%D1%8F%D1%82%D0%BD%D0%B8%D0%BA%D0%B8_%D0%B0%D1%80%D1%85%D0%B8%D1%82%D0%B5%D0%BA%D1%82%D1%83%D1%80%D1%8B_%D0%9C%D0%BE%D1%81%D0%BA%D0%B2%D1%8B"
        r = requests.get(url)
        soup = BeautifulSoup(r.text, features="lxml")
        f = False
        link_sp = []
        name_list=[]
        lat_list=[]
        lon_list=[]
        count = 1
        for i in soup.find_all('div', class_="mw-category-group"):
            for j in i.find_all('a'):
                url1 = 'https://ru.wikipedia.org' + j['href']
                if count > 4:
                    r = requests.get(url1)
                    link_sp.append(url1)
                    soup = BeautifulSoup(r.text, features="lxml")
                    # Ниже использую регулярные выражения, так как без их использования выходит сложное решение
                    s = r.text
                    if soup.find(class_='mw-kartographer-maplink') != None:
                        ps = re.findall(r'data-lat=(.*?)\n', s)
                        ps1 = re.findall(r'data-lon=(.*?)\n', s)
                        lat = float(re.sub("[^.0-9]", "", ps[0][0:12]))
                        lon = float(re.sub("[^.0-9]", "", ps1[0][0:12]))
                        lat_list.append(lat)
                        lon_list.append(lon)
                        name_list.append(j.text)
                count = count + 1
        df_wiki = pd.DataFrame(list(zip(name_list, lon_list, lat_list)),
                       columns =['name', 'long', 'lat'])
        df_wiki.to_csv(r'D:\FinalProject\parsed_data.csv', index=False)
    ###END PARSING
    ###ИЗВЛЕЧЕНИЕ ДАННЫХ ИЗ БОЛЬШОГО ФАЙЛА ПО РОССИИ В ФАЙЛ С ДАННЫМИ ПО МОСКВЕ
    def parsing_home():
        df_parse = read_file("all_v2.csv")
        df_parse[df_parse['region'] == 3].to_csv(r'D:\FinalProject\moscowhome_data.csv', index=False)
    #ЭТОЙ ФУНКЦИЕЙ Я НАХОЖУ ДИСТАНЦИИ ОТ НЕДВИЖИМОСТИ ДО ЦЕНТРА МОСКВЫ
    def distance_get():
        dist_spis = []
        df_moscow['dist'] = geopy.distance.geodesic((55.7522, 37.6156), (54.7522, 36.6156)).km
        for i in range(len(df_moscow)):
            coords_1 = (55.7522, 37.6156)
            coords_2 = (df_moscow.geo_lat[i], df_moscow.geo_lon[i])
            res = geopy.distance.geodesic(coords_1, coords_2).km
            dist_spis.append(res)
        df_moscow.to_csv(r'D:\FinalProject\moscowdist_data.csv', index=False)
        return (dist_spis)

    st.title('Final Project')
    st.subheader('Мой проект должен быть полезным для тех, кто собирается переселяться в Москву и для новых туристов.')
    st.subheader('Сначала сделаем анализ по данным достопримечательностей из Википедии:')
    """
    В первую очередь я спарсил данные с Википедии по достопримечательностям.
    Оттуда я взял их список и координаты, переходя по каждой ссылке по-отдельности. Парсинг проходил отдельно от всей программы,
    чтобы streamlit долго не грузился.
    Функция парсинга: def parsing_mem(), использовал в ней регулярные выражения, которые сильно упростили написание
    функции.
    Ссылка на объект парсинга: https://ru.wikipedia.org/wiki/Категория:Памятники_архитектуры_Москвы
    """
    # Парсинг
    df_dost = read_file("parsed_data.csv")
    df_dost['dist'] = 0
    df_dost['dist_center'] = 0
    for i in range(len(df_dost)):
        coords_1 = (df_dost.lat[i], df_dost.long[i])
        coords_2 = (55.7522, 37.6156)
        df_dost.dist_center[i] = geopy.distance.geodesic(coords_1, coords_2).km
    st.write(df_dost)
    df1 = df_dost

    with open('moscow_polygon.geojson', encoding = 'utf-8') as f1:
        b = json.load(f1)
    moscow_poly = Polygon(b['features'][0]['geometry']['coordinates'][0])

    """
    С помощью этого функционала вы можете посмотреть, какие 5 достопримечтательностей самые близкие к вам. 
    """
    place_lat = st.number_input('Введите latitude для того, чтобы найти 5 ближайших объектов для пешей прогулки')
    place_long = st.number_input('Введите longitude для того, чтобы найти 5 ближайших объектов для пешей прогулки')

    point = Point(place_long, place_lat)
    if moscow_poly.contains(point) == True:
        st.success("Поздравляю, вы в Москве, вот ближайшие 5 точек к вам (вы - это синяя точка на карте)")
        for i in range(len(df_dost)):
            coords_1 = (df_dost.lat[i], df_dost.long[i])
            coords_2 = (place_lat, place_long)
            df_dost.dist[i] = geopy.distance.geodesic(coords_1, coords_2).km
            df_dost.dist_center[i] = geopy.distance.geodesic(coords_1, (55.7522, 37.6156)).km
        sorted_df = df_dost.sort_values(by='dist')
        st.write(sorted_df[:5])
        f = True
    else:
        st.error("Вы не в Москве, введите занова координаты, учитывая ее центр 55,7522:37,6156")
        f = False
    """
    Далее я построил тепловую карту, на которой отображаются все основные достопримечательности.
    После ввода ваших координат, вы будете отмечены синей точкой на карте.
    """
    m_2 = folium.Map(
        location=[55.7522, 37.6220],
        tiles='Stamen Toner',
        zoom_start=12
    )
    for i in range(len(df_dost)):
        folium.Circle(
            radius=40,
            location=[df_dost.lat[i], df_dost.long[i]],
            popup=df_dost.name[i],
            color='crimson',
            fill=True,
        ).add_to(m_2)
    HeatMap(df_dost[['lat', 'long']], radius=20).add_to(m_2)
    if f == True:
        folium.Circle(
                radius=60,
                location=[place_lat, place_long],
                popup="Здесь Вы",
                color='blue',
                fill=False,
                fill_color = True,
            ).add_to(m_2)
    folium_static(m_2)
    """
    Дальше я построил violine график распределения достопримечательностей по удаленности от центра,
    он явно визуализирует медиану, среднее и другие данные статистики.
    """
    #GRAPH 1
    df_2 = df_dost
    fig2 = px.violin(df_2, y="dist_center", box=True,
                     labels={'country': 'Country', 'lon': 'Longitude', 'lat': 'Latitude',
                             'suicides_per_100k_pop': 'Number of suicides per 100k'},
                     points='all',
                     title="Violin график, показывающий распределение достопримечательностей по отдаленности от центра ",
                     )
    st.plotly_chart(fig2)


    st.subheader('Дальше сделаем анализ по датабазе недвижимости:')
    """
    Для того, чтобы получить данные по Москве, мне пришлось из большого файла all_v2.csv по недвижимости в России
    вытащить лишь искомые. Этот процесс происходит в функции def parsing_home(), которая была выключена из-за 
    долгого времени работы streamlit. 
    """
    df_moscow = read_file("moscowhome_data.csv")

    df_moscow1 = read_file("moscowdist_data.csv")
    df_moscow1['price_meter'] = df_moscow1['price']/df_moscow1['area']
    df_moscow1.replace({'rooms':{-1:0}}, inplace=True)
    df_moscow1['price_range'] = 0
    """
    Также на данном этапе я дополняю таблицу новым столбцом характеристик недвижимости - уровнем цен. Для этого я использую
    NUMPY, поскольку это сильно упрощает работу.
    """
    #ИСПОЛЬЗУЮ NUMPY КАК СПОСОБ БЫСТРОЙ ЗАМЕНЫ В ТАБЛИЦЕ РАДИ ЭКОНОМИИ ВРЕМЕНИ
    df_moscow1['price_range'] = np.where((df_moscow1.price < 20000000), '10-20mln', df_moscow1.price_range)
    df_moscow1['price_range'] = np.where((df_moscow1.price < 10000000), '5-10mln', df_moscow1.price_range)
    df_moscow1['price_range'] = np.where((df_moscow1.price < 5000000), '< 5mln', df_moscow1.price_range)
    df_moscow1['price_range'] = np.where((df_moscow1.price >= 20000000), '> 20mln', df_moscow1.price_range)

    st.write(df_moscow1)
    df2 = df_moscow1
    """
    Теперь я строю график, который визуализирует распределения по количествам
    объявлений на продажу в зависимости от характеристик.
    """
    ##FIRST GRAPH
    groupby_column = st.selectbox(
        'Пожалуйста, выберите категорию, по которой вы хотите получить анализ:',
        ('building_type', 'rooms', 'price_range')
    )

    # GROUP DATAFRAME 1
    df_moscow1['nomb'] = 1
    df_sum = df_moscow1.groupby(by=[groupby_column], as_index=False)['nomb'].sum()

    # PLOT DATAFRAME 1
    # FROM https://www.youtube.com/watch?v=ZDffoP6gjxc&t=116s
    if groupby_column == 'rooms':
        """
        Значение 0 в разряде комнат означает студии.
        """
    if groupby_column == 'building_type':
        """
         Обозначения такие::
         0 - Другой. 1 - Панельный. 2 - Монолитный. 3 - Кирпичный. 4 - Блочный. 5 - Деревянный
         """
    fig4 = px.bar(
        df_sum,
        x=groupby_column,
        y="nomb",
        labels={'nomb': 'Число объявлений'},
        color='nomb',
        color_continuous_scale=['green', 'yellow', 'red'],
        template='presentation',
        title=f'<b> Число объявлений, сгруппированных по {groupby_column} </b>'
    )
    st.plotly_chart(fig4)

    # END FROM
    """
    Далее я строю временной график выдачи объявлений о продаже недвижимости в Москве.
    По нему можно проанализировать сезонность продаж недвижимости и сделать выводы о том, когда
    новому жителю выгоднее всего купить себе квартиру.
    """
    #GRAPH 3 TIME
    df_grouped = df_moscow1.groupby(by=['date'], as_index=False)['nomb'].sum()
    fig3 = go.Figure([go.Scatter(x=df_grouped['date'], y=df_grouped['nomb'])])
    fig3.update_xaxes(title_text='Дни')
    fig3.update_yaxes(title_text='Число заявок на продажу')
    st.plotly_chart(fig3)

    #
    #NOW GEOPANDAS ВОЗЬМЕМ ПОЛИГОНЫ РАЙОНОВ МОСКВЫ ИЗ ДОМАШКИ

    st.subheader('Посмотрим, какой график получится по районам в GeoPandas:')
    '''
    Первоначально я беру файлы moscow_polygon.geojson и преобразую его в adm_moscow посредством нижеидущих операций.
    Основная проблема возникла при выгрузке geopandas в streamlit, поскольку он не предусматривает 
    постройку карт geopandas на платформе. 
    '''
    '''
    Поэтому ниже я оставляю код своей программы, которая строит данную карту, чтобы вы могли изучить
    и убедиться в ее работе. Саму карту я строил в Jupiter.
    '''
    ''' 
    МОЙ КОД:
    '''
    '''
    df_moscow1 = read_file("Jupiter_data.csv")
    gdf4 = gpd.GeoDataFrame(df_moscow1, geometry = gpd.points_from_xy(df_moscow1['geo_lon'], df_moscow1['geo_lat']))
    gdf_map = gpd.sjoin(gdf4, gdf3, predicate = "intersects", how = 'right')
    gdf_map['name'].replace(regex=True,inplace=True,to_replace=' район',value='') 
    gdf_map['name'].replace(regex=True,inplace=True,to_replace='район ',value='') 
    number = gdf_map['name'].value_counts()
    print(number)
    adm_moscow = gpd.read_file("http://gis-lab.info/data/mos-adm/mo.geojson")
    %matplotlib inline
    import matplotlib.pyplot as plt
    adm_moscow = adm_moscow.set_index("NAME").assign(number_counts=number)
    print(adm_moscow)
    fig, ax = plt.subplots(figsize=(10, 10))
    adm_moscow.plot(column = "number_counts", ax=ax, legend = True)
    '''
    image = Image.open('map.jpg')
    st.image(image, caption='Sunrise by the mountains')





    print('FINE')

    st.subheader('Загрузки:')
    """
    Дополнительной возможностью из тех, что мы не изучали является моя работа с base64 и io. Однако данных библиотек нет
    при выгрузке в streamlit, поэтому вы можете посмотреть данный функционал только при запуске кода, что я прошу сделать. 
    С помощью данной функции вы можете скачать все графики и все таблицы, которые представлены в моем проекте. Функции
    def generate_excel_download_link; def generate_html_download_link
    """

    """EXCEL FILES for 1 and 2 tablets"""
    "1"
    generate_excel_download_link(df1)
    "2"
    generate_excel_download_link(df1)

    """HTML FILES for 1, 2, 3 graphs"""
    "1"
    generate_html_download_link(fig2)
    "2"
    generate_html_download_link(fig4)
    "3"
    generate_html_download_link(fig3)


