import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(10)
import pandas as pd
import seaborn as sns
import time
from sklearn import decomposition
from sklearn import preprocessing
from matplotlib.collections import LineCollection
from scipy.stats import norm
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import metrics 
from sklearn.metrics import confusion_matrix
import statsmodels.api as sm
import scipy
import contextlib
import urllib

sns.set_theme(style="whitegrid")

PALETTE = {"a":"green",
           "b":"yellowgreen", 
           "c":"yellow",
           "d":"lightsalmon",
           "e":"red"}

data_url = 'https://www.googleapis.com/drive/v3/files/1KM_A8-FDwhb3sNUOJVI-Oa3_7uYYnlyG?alt=media&key=AIzaSyA02kIHmvFlznhF6cPjlQYx28Ke-4OU68k'

@st.cache(suppress_st_warning=True)
def load_data():
    chunksize = 10 ** 4
    list_of_dfs = []
    my_bar= st.progress(0)
    
    with contextlib.closing(urllib.request.urlopen(url=data_url)) as rd:
        for number, chunk in enumerate(pd.read_csv(rd, chunksize=chunksize, index_col=0)):
            list_of_dfs.append(chunk)
            my_bar.progress(number+5)
    
    df=pd.concat(list_of_dfs)
    my_bar = my_bar.empty()
    return df

def kde_plot(nutriment, df):
    fig, ax = plt.subplots()
    sns.kdeplot(data=df, x=nutriment, hue='nutriscore_grade', cut=0, bw_adjust=1, palette=PALETTE ,hue_order = ['a', 'b','c','d','e'], linewidth=2)
    return fig

def boxplots(nutriment, df):
    fig, ax = plt.subplots()
    sns.boxplot(x=df[nutriment], y=df['nutriscore_grade'], palette=PALETTE, order=['a', 'b','c','d','e'], showfliers = False)
    return fig

def pnns_group(df):
    fig, ax = plt.subplots()
    ax = sns.barplot(y=df["pnns_groups_1"].value_counts().sort_values(ascending=False).index, x=df["pnns_groups_1"].value_counts().sort_values(ascending=False).values)
    for p in ax.patches:
        _x = p.get_x() + p.get_width() + float(400)
        _y = p.get_y() + p.get_height() - float(0.3) 
        value = str( round(int(p.get_width()) / len(df) * 100,1) ) + '%'
        ax.text(_x, _y, value, ha="left")
    plt.xlabel("Nombre de produits")
    return fig

def produits_par_nutrigrade(df):
    produitsParNutrigrade=df['nutriscore_grade'].value_counts().sort_values(ascending=False)
    fig, ax = plt.subplots()
    my_circle = plt.Circle((0,0), 0.6, color='white')
    my_pie, _, _ = plt.pie(produitsParNutrigrade.values, labels=produitsParNutrigrade.index, autopct="%.1f%%", pctdistance=0.8,colors=["lightsalmon","yellow","green","yellowgreen","red"])
    p = plt.gcf()
    p.gca().add_artist(my_circle)
    return fig

def pnns_nutrigrade(df):
    plotdata=pd.crosstab(df.pnns_groups_1,df.nutriscore_grade)
    stacked_data = plotdata.apply(lambda x: x*100/sum(x), axis=1)
    fig, ax = plt.subplots()
    stacked_data.plot(kind="bar", stacked=True, ax=ax, color=PALETTE, width=0.9,alpha=0.8)
    plt.legend(loc='upper right', bbox_to_anchor=(1.12, 1.01));
    plt.xlabel("")
    plt.ylabel("%")
    return fig

def produits_par_pays(df):
    produitsParPays=df['countries_en'].value_counts().sort_values(ascending=False)
    top5produitParPays=df['countries_en'].value_counts().sort_values(ascending=False).head(5)
    top5produitParPays=top5produitParPays.append(pd.Series([produitsParPays.iloc[5:].sum()], index=['Other countries'])).sort_values(ascending=False)
    fig, ax = plt.subplots()
    my_circle = plt.Circle((0,0), 0.6, color='white')
    my_pie, _, _ = plt.pie(top5produitParPays.values, labels=top5produitParPays.index, autopct="%.1f%%", pctdistance=0.8)
    p = plt.gcf()
    p.gca().add_artist(my_circle)
    return fig

def nutrigrade_par_brand(df):
    df_with_brand=df.dropna(subset=['brands','nutriscore_grade'])
    topBrandsList=df_with_brand.brands.value_counts().head(10).index
    topBrands=df_with_brand[df_with_brand['brands'].isin(topBrandsList)]
    cont=pd.crosstab(topBrands['brands'], topBrands['nutriscore_grade'])
    stacked_data = cont.apply(lambda x: x*100/sum(x), axis=1)

    fig, ax = plt.subplots()
    stacked_data.plot(kind="bar", stacked=True, ax=ax, color=PALETTE, width=0.9,alpha=0.8)
    plt.legend(loc='upper right', bbox_to_anchor=(1.12, 1.01));
    plt.xlabel("")
    plt.ylabel("%")
    return fig

@st.cache()
def fit_pca(df):
    n_comp = 6
    data_pca = df[['energy-kcal_100g','fat_100g','carbohydrates_100g','sugars_100g','proteins_100g','salt_100g','saturated-fat_100g']]
    X = data_pca.values
    names = df['nutriscore_grade']
    features = data_pca.columns

    std_scale = preprocessing.StandardScaler().fit(X)
    X_scaled = std_scale.transform(X)

    pca = decomposition.PCA(n_components=n_comp)
    pca.fit(X_scaled)
    pcs = pca.components_
    return pca, pcs, X_scaled, n_comp, features

def display_scree_plot(pca):
    fig, ax = plt.subplots()
    scree = pca.explained_variance_ratio_*100
    plt.bar(np.arange(len(scree))+1, scree)
    plt.plot(np.arange(len(scree))+1, scree.cumsum(),c="red",marker='o')
    plt.xlabel("rang de l'axe d'inertie")
    plt.ylabel("pourcentage d'inertie")
    return fig

def display_circles(pcs, n_comp, pca, axis_ranks, labels=None, label_rotation=0, lims=None):
    for d1, d2 in axis_ranks: # On affiche les 3 premiers plans factoriels, donc les 6 premières composantes
        if d2 < n_comp:

            # initialisation de la figure
            fig, ax = plt.subplots()

            # détermination des limites du graphique
            if lims is not None :
                xmin, xmax, ymin, ymax = lims
            elif pcs.shape[1] < 30 :
                xmin, xmax, ymin, ymax = -1, 1, -1, 1
            else :
                xmin, xmax, ymin, ymax = min(pcs[d1,:]), max(pcs[d1,:]), min(pcs[d2,:]), max(pcs[d2,:])

            # affichage des flèches
            # s'il y a plus de 30 flèches, on n'affiche pas le triangle à leur extrémité
            if pcs.shape[1] < 30 :
                plt.quiver(np.zeros(pcs.shape[1]), np.zeros(pcs.shape[1]),
                   pcs[d1,:], pcs[d2,:], 
                   angles='xy', scale_units='xy', scale=1, color="grey")
                # (voir la doc : https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.html)
            else:
                lines = [[[0,0],[x,y]] for x,y in pcs[[d1,d2]].T]
                ax.add_collection(LineCollection(lines, axes=ax, alpha=.1, color='black'))
            
            # affichage des noms des variables  
            if labels is not None:  
                for i,(x, y) in enumerate(pcs[[d1,d2]].T):
                    if x >= xmin and x <= xmax and y >= ymin and y <= ymax :
                        plt.text(x, y, labels[i], fontsize='14', ha='center', va='center', rotation=label_rotation, color="blue", alpha=0.5)
            
            # affichage du cercle
            circle = plt.Circle((0,0), 1, facecolor='none', edgecolor='b')
            plt.gca().add_artist(circle)

            # définition des limites du graphique
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
        
            # affichage des lignes horizontales et verticales
            plt.plot([-1, 1], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-1, 1], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Cercle des corrélations (F{} et F{})".format(d1+1, d2+1))
    return fig

def display_factorial_planes(X_projected, n_comp, pca, axis_ranks, labels=None, alpha=1, illustrative_var=None, palette=None):
    for d1,d2 in axis_ranks:
        if d2 < n_comp:
 
            # initialisation de la figure       
            fig = plt.figure()
        
            # affichage des points
            if illustrative_var is None:
                plt.scatter(X_projected[:, d1], X_projected[:, d2], alpha=alpha)
            else:
                illustrative_var = np.array(illustrative_var)
                for value in np.unique(illustrative_var):
                    col=palette[value]
                    selected = np.where(illustrative_var == value)
                    plt.scatter(X_projected[selected, d1], X_projected[selected, d2], color=col, alpha=alpha, label=value)
                plt.legend()

            # affichage des labels des points
            if labels is not None:
                for i,(x,y) in enumerate(X_projected[:,[d1,d2]]):
                    plt.text(x, y, labels[i],
                              fontsize='14', ha='center',va='center') 
                
            # détermination des limites du graphique
            boundary = np.max(np.abs(X_projected[:, [d1,d2]])) * 1.1
            plt.xlim([-boundary,boundary])
            plt.ylim([-boundary,boundary])
        
            # affichage des lignes horizontales et verticales
            plt.plot([-100, 100], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-100, 100], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))
    return fig

def fit_norm(nutriment,df):
    mu, std = norm.fit(df[nutriment])
    fig = plt.figure(figsize=(7,6))
    plt.hist(df[nutriment], bins=10, density=True, alpha=0.6, color='slateblue')
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    return fig

def prob_plot(nutriment, df):
    fig = plt.figure()
    res = stats.probplot(df[nutriment], plot=plt)
    return fig

def shapiro_test(df, alpha = 0.05):
    stats_res = []
    ps = []
    reject =[]
    for column in df.columns:
        stat, p = stats.shapiro(df[column])
        if p > alpha:
            reject.append('normal')
        else:
            reject.append('not normal')
        stats_res.append(stat)
        ps.append(p)
    res = pd.DataFrame({'stats': stats_res, 'p-value': ps, 'distribution': reject}, index=df.columns)
    return res

def kruskal_for_many_var(df, x, alpha = 0.05, y='nutriscore_grade'):
    stats_res = []
    ps = []
    reject =[]
    for variable in x:
        stat, p = stats.kruskal(*[group[variable].values for name, group in df.groupby(y)])
        if p < alpha:
            reject.append('significant')
        else:
            reject.append('not significant')
        stats_res.append(stat)
        ps.append(p)
    res = pd.DataFrame({'stats': stats_res, 'p-value': ps, 'result': reject}, index=x)
    return res

@st.cache()
def fit_multinom_logit(df, subset):
    df_sample=df.dropna(subset=['nutriscore_grade']).sample(10000)
    y=df_sample['nutriscore_grade']
    x=df_sample[subset]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, stratify=y,random_state = 5)
    logit_model=sm.MNLogit(y_train, X_train)
    result=logit_model.fit()
    stats=result.summary()
    return stats

def chi2_test(df):
    df_with_brand=df.dropna(subset=['brands','nutriscore_grade'])
    topBrandsList=df_with_brand.brands.value_counts().head(10).index
    topBrands=df_with_brand[df_with_brand['brands'].isin(topBrandsList)]
    topBrandsSample=topBrands.sample(1000)
    cont=pd.crosstab(topBrandsSample['brands'], topBrandsSample['nutriscore_grade'])
    st_chi2, st_p, st_dof, st_exp=scipy.stats.chi2_contingency(cont)
    chi2=pd.DataFrame({'stats':[st_chi2],'p-value':[st_p],'degree of freedom':[st_dof], 'res':['significatif']})
    return cont, chi2

def app():

    st.set_page_config(layout="wide")
    menu = ["Home", "Visualisations", "ACP", "Tests des hyppothèses"]
    st.sidebar.title("Menu")
    choice = st.sidebar.selectbox("",menu)
    df = load_data()
    
    if choice == "Visualisations":
        
        nutriments = ('fat_100g','carbohydrates_100g','sugars_100g','proteins_100g','salt_100g','saturated-fat_100g','energy-kcal_100g')
        nutriment = st.selectbox("Choix du nutriment", nutriments)        
        st.subheader(f"Distribution et boxplot de {nutriment} en fonction du nutri-score")

        col1, col2 = st.beta_columns(2)
        
        with col1:
            with st.beta_container():
                st.write("")
                st.pyplot(kde_plot(nutriment, df))

        with col2:
            with st.beta_container():
                st.write("")
                st.pyplot(boxplots(nutriment, df))

        with st.beta_container():
            st.subheader('Nombre de produits par PNNS groupe')
            st.write("")
            st.pyplot(pnns_group(df))

        col3, col4 = st.beta_columns(2)

        with col3:
            with st.beta_container():
                st.subheader('Répartition des produits dans la base de donnée en fonction de leur nutri-score')
                st.write("")
                st.pyplot(produits_par_nutrigrade(df))
               
        with col4:
            with st.beta_container():
                st.subheader('Répartition des nutri-scores au sein des catégories établies par PNNS')
                st.write("")
                st.pyplot(pnns_nutrigrade(df))
        
        col5, col6 = st.beta_columns(2)
        
        with col5:
            with st.beta_container():
                st.subheader('Répartition des nutri-scores parmi les produits commercialisés par les 10 plus grandes marques')
                st.write("")
                st.pyplot(nutrigrade_par_brand(df))

        with col6:
            with st.beta_container():
                st.subheader('Pourcentage de produits par pays')
                st.write("")
                st.pyplot(produits_par_pays(df))

    elif choice == "ACP":
        
        col7, col8 = st.beta_columns(2)
        pca, pcs, X_scaled, n_comp, features = fit_pca(df)
        
        with col7:
            with st.beta_container():
                st.subheader('Eboulis des valeurs propres')
                st.write("")
                st.pyplot(display_scree_plot(pca))

        with col8:
            with st.beta_container():
                st.subheader('Cercle des corrélations')
                st.write("")
                st.pyplot(display_circles(pcs, n_comp, pca, [(0,1)], labels = np.array(features)))

        with st.beta_container():
                X_projected = pca.transform(X_scaled)
                st.subheader('Projection des individus sur le premier et deuxième plan factoriel')
                st.write("")
                st.pyplot(display_factorial_planes(X_projected[:2000], n_comp, pca, [(0,1)], illustrative_var = df['nutriscore_grade'].dropna()[:2000], palette=PALETTE, alpha=0.5))

    elif choice == "Tests des hyppothèses":
        st.subheader('Hypothèse 1a: les produits avec un nutri-score différent ne contiennent pas la même quantité de sucre / gras / gras saturé / kilocalories / carbohydrate / proteins / sel.')
        st.text(""" * Vérifier la distribution de données
* Confirmer avec un test Shapiro-Wilk
* Véfifier l'hyppothèse avec un test Kruskal-Wallis""")

        nutriments = ['energy-kcal_100g','fat_100g','carbohydrates_100g','sugars_100g','proteins_100g','salt_100g','saturated-fat_100g']
        nutriment = st.selectbox("Choix du nutriment", nutriments)        
        st.subheader(f"Distribution empirique de {nutriment} et sa distribution théorique normale")
        
        col9, col10 = st.beta_columns(2)

        with col9:
                with st.beta_container():
                    st.write("")
                    st.pyplot(fit_norm(nutriment,df))
        
        with col10:
                with st.beta_container():
                    st.write("")
                    st.pyplot(prob_plot(nutriment, df))

        col11, col12 = st.beta_columns(2)

        with col11:
            with st.beta_container():
                st.subheader('Test Shapiro-Wilk')
                normRes=shapiro_test(df[nutriments].sample(1000))
                normStyle=normRes.style.format(formatter={'p-value': "{:.3g}"})
                st.dataframe(normStyle)
        
        with col12:
            with st.beta_container():
                st.subheader('Test Kruskal-Wallis')
                kruskalRes=kruskal_for_many_var(df.sample(1000), x=nutriments)
                kruskalStyle=kruskalRes.style.format(formatter={'p-value': "{:.3g}"})
                st.dataframe(kruskalStyle)

        st.subheader('Hypothèse 1b: changement du quantite de sucre / gras / gras saturé / kilocalories / carbohydrate / proteins / sel dans un produit influence son nutri-score.')
        stats=fit_multinom_logit(df, nutriments)
        st.text(stats)
        st.subheader("L'hyppothèse 2: l’existence d’un lien entre la marque et le nutri-score des produits qu'elle commercialise")
        st.text('Chi2 est le test le plus souvent utilisé pour tester un lien entre les deux variables catégorielles')

        cont, chi2 = chi2_test(df)
        
        st.dataframe(cont)
        st.write("")
        chi2style=chi2.style.format({'p-value': "{:.3g}"}).hide_index()
        st.dataframe(chi2style)

    else:
        st.title("Exploration de la base de données Open Food Facts")
        st.markdown("👈 **Les différents étapes de travail.**")
        col13, col14 = st.beta_columns(2)
        with col13:
            with st.beta_container():
                st.image("https://upload.wikimedia.org/wikipedia/fr/thumb/5/50/Sante-publique-France-logo.svg/1200px-Sante-publique-France-logo.svg.png")

        with col14:
            with st.beta_container():
                st.markdown('''<div><p> L'agence Santé publique France souhaite rendre les données de santé publique plus accessibles, \
                     pour qu’elles soient utilisables par ses agents. Pour cela, nous faisons appel à vous pour réaliser une \
                         première exploration et visualisation des données, afin que nos agents puissent ensuite s’appuyer sur vos résultats. </p>\

                        <p> L'analyse est basée sur le jeu de données Open Food Facts. \
                            Les champs sont séparés en quatre sections :\
                            <ul>
                                <li>Les informations générales sur la fiche du produit : nom, date de modification, etc. </li>\
                                <li>Un ensemble de tags : catégorie du produit, localisation, origine, etc.</li>\
                                <li>Les ingrédients composant les produits et leurs additifs éventuels. </li>\
                                <li>Des informations nutritionnelles : quantité en grammes d’un nutriment pour 100 grammes du produit. </li>
                            </ul> </p></div>
                        ''', unsafe_allow_html=True)
if __name__ == "__main__":
    app()