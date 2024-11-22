import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
# Carregar o dataset
df = pd.read_csv('Churn_Modelling.csv')
# Configurar estilo e paleta de cores primárias
sns.set_style("whitegrid")
custom_palette = ['#FF0000', '#0000FF', '#FFFF00']  # Vermelho, Azul, Amarelo

# Gráfico 1: Distribuição de Clientes (Exited)
plt.figure()
sns.countplot(x='Exited', data=df, palette=custom_palette[:2]) 
plt.title("Distribuição de Clientes (Exited)")
plt.xlabel("Exited (0 = Ficou, 1 = Saiu)")
plt.ylabel("Contagem")
plt.savefig('clientes_sair.png', dpi=300, bbox_inches='tight')
plt.show()

# Gráfico 2: Distribuição de Clientes por Gênero
plt.figure()
sns.countplot(x='Gender', data=df, palette=custom_palette[:2])
plt.title("Distribuição de Clientes (Gênero)")
plt.xlabel("Gênero")
plt.ylabel("Contagem")
plt.savefig('clientes_genero.png', dpi=300, bbox_inches='tight')
plt.show()

# Gráfico 3: Distribuição de Clientes por Países
plt.figure()
sns.countplot(x='Geography', data=df, palette='coolwarm') 
plt.title("Distribuição de Clientes (Países)")
plt.xlabel("Geography")
plt.ylabel("Contagem")
plt.savefig('clientes_paises.png', dpi=300, bbox_inches='tight')
plt.show()

# Gráfico 4: Taxa de Churn por País
country_churn = df.groupby('Geography')['Exited'].mean().reset_index()
country_churn.columns = ['Country', 'Churn Rate']
plt.figure()
sns.barplot(x='Country', y='Churn Rate', data=country_churn, palette='coolwarm')
plt.title('Taxa de Churn por País')
plt.xlabel('País')
plt.ylabel('Taxa de Churn')
plt.savefig('churn_por_pais.png', dpi=300, bbox_inches='tight')
plt.show()

# Gráfico 5: Taxa de Churn por Gênero
gender_churn = df.groupby('Gender')['Exited'].mean().reset_index()
gender_churn.columns = ['Gênero', 'Churn Rate']
plt.figure()
sns.barplot(x='Gênero', y='Churn Rate', data=gender_churn, palette=custom_palette[:2]) 
plt.title('Taxa de Churn por Gênero')
plt.xlabel('Gênero')
plt.ylabel('Taxa de Churn')
plt.savefig('churn_por_genero.png', dpi=300, bbox_inches='tight')
plt.show()

# Gráfico 6: Taxa de Churn por Faixa Etária
bins = [18, 25, 35, 45, 55, 65, 75]
labels = ['18-24', '25-34', '35-44', '45-54', '55-64', '65-74']
df['Faixa Etária'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
age_churn = df.groupby('Faixa Etária')['Exited'].mean().reset_index()
age_churn.columns = ['Faixa Etária', 'Churn Rate']
plt.figure()
sns.barplot(x='Faixa Etária', y='Churn Rate', data=age_churn, palette='coolwarm')
plt.title('Taxa de Churn por Faixa Etária')
plt.xlabel('Faixa Etária')
plt.ylabel('Taxa de Churn')
plt.savefig('churn_por_faixa_etaria.png', dpi=300, bbox_inches='tight')
plt.show()

# Gráfico 7: Taxa de Churn por País e Gênero
grouped_data = df.groupby(['Geography', 'Gender'])['Exited'].mean().reset_index()
grouped_data.columns = ['Geography', 'Gender', 'Churn Rate']
plt.figure()
sns.barplot(x='Geography', y='Churn Rate', hue='Gender', data=grouped_data, palette=custom_palette[:2])
plt.title("Taxa de Churn por País e Gênero")
plt.xlabel("País")
plt.ylabel("Taxa de Churn")
plt.savefig('churn_por_pais_genero.png', dpi=300, bbox_inches='tight')
plt.show()

# Gráfico 8: Heatmap - Faixa Etária vs Faixa de Saldo
bins_balance = [0, 50000, 100000, 150000, 200000, 250000, 300000]
labels_balance = ['0-50k', '50k-100k', '100k-150k', '150k-200k', '200k-250k', '250k+']
df['Faixa de Saldo'] = pd.cut(df['Balance'], bins=bins_balance, labels=labels_balance, right=False)
count_data = df.groupby(['Faixa Etária', 'Faixa de Saldo']).size().reset_index(name='Contagem')
heatmap_data = count_data.pivot(index='Faixa Etária', columns='Faixa de Saldo', values='Contagem').fillna(0)
plt.figure(figsize=(12, 8))
sns.heatmap(
    heatmap_data,
    annot=True, fmt=".0f", cmap='coolwarm', cbar_kws={'label': 'Número de Clientes'}
)
plt.title('Distribuição de Clientes por Faixa Etária e Saldo Bancário')
plt.xlabel('Faixa de Saldo Bancário')
plt.ylabel('Faixa Etária')
plt.savefig('heatmap_faixa_etaria_saldo.png', dpi=300, bbox_inches='tight')
plt.show()


# Gráfico 9: Evasão por saldo
bins = [0, 50000, 100000, 150000, 200000, 250000, df['Balance'].max()]
labels = ['Até 50k', '50k-100k', '100k-150k', '150k-200k', '200k-250k', '250k+']
df['Faixa de Saldo'] = pd.cut(df['Balance'], bins=bins, labels=labels, right=False)
saldo_churn = df.groupby('Faixa de Saldo')['Exited'].mean().reset_index()
saldo_churn.columns = ['Faixa de Saldo', 'Taxa de Evasão']
plt.figure(figsize=(10, 6))
sns.barplot(x='Faixa de Saldo', y='Taxa de Evasão', data=saldo_churn, palette='coolwarm')
plt.title('Taxa de Evasão por Faixa de Saldo')
plt.xlabel('Faixa de Saldo Bancário')
plt.ylabel('Taxa de Evasão')
plt.xticks(rotation=45)
plt.savefig('evasao_por_saldo.png', dpi=300, bbox_inches='tight')
plt.show()


# Gráfico 9: Evasão por cliente ativo
active_churn = df.groupby('IsActiveMember')['Exited'].mean().reset_index()
active_churn.columns = ['Membro Ativo', 'Taxa de Evasão']
active_churn['Membro Ativo'] = active_churn['Membro Ativo'].map({0: 'Inativo', 1: 'Ativo'})
plt.figure(figsize=(8, 5))
sns.barplot(x='Membro Ativo', y='Taxa de Evasão', data=active_churn, palette=custom_palette[:2])
plt.title('Taxa de Evasão por Status de Atividade')
plt.xlabel('Status de Atividade')
plt.ylabel('Taxa de Evasão')
plt.savefig('evasao_por_status_ativo.png', dpi=300, bbox_inches='tight')
plt.show()