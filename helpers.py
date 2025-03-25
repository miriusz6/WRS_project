import pandas as pd
import numpy as np  

def clean_data(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.astype({'user_id': 'str', 'item_id': 'str', 'rating': 'float'}, copy=False)
    df.drop(['timestamp'], axis=1, inplace=True)
    print("Data points initially:", df.shape[0])
    df.drop_duplicates(inplace=True, keep='first')
    print("Data points after removing duplicates:", df.shape[0])
    
    df.dropna(inplace=True)

    # invalid user_id or item_id
    t1 = df.loc[df['item_id'] == ''].shape[0] + df.loc[df['user_id'] == ''].shape[0]
    assert t1 == 0
    print("No invalid user_id or item_id")
    # invalid ratings
    t2 = df.loc[df['rating'] < 0].shape[0] + df.loc[df['rating'] > 5].shape[0]  
    assert t2 == 0
    print("No invalid ratings")

    print("Data points after cleaning:", df.shape[0])

def read_N_prepare_data():
    # read data
    df_t = pd.read_csv('data/test.tsv', sep='\t')
    df = pd.read_csv('data/train.tsv', sep='\t')
    # clean data
    print("TEST DATA:")
    clean_data(df_t)
    print("TRAINING DATA:")
    clean_data(df)
    # check if all users in test data are in training data
    users = frozenset(df['user_id'])
    users_t = frozenset(df_t['user_id'])
    t1 = users_t.issubset(users)
    assert t1 is True
    if t1:
        print( "All users in test data are in training data")
    return df, df_t
    

# def plot_item_ratings_freq(df):
#     # item ratings overview
#     rating_cnts = df['item_id'].value_counts()
#     print(rating_cnts.head(10))
#     print(rating_cnts.tail(10))
#     plot_kwargs = {
#         'kind':'bar',
#         'title':'Item ratings count',
#         'figsize':(10,5),
#         'xticks':range(0, rating_cnts.shape[0], 10),
#         'use_index':False,
#         'grid':True
#         }
#     # plot item ratings
#     ax = rating_cnts.plot(**plot_kwargs)
#     ax.set_xlabel('Sorted item index')
#     ax.set_ylabel('Ratings count')
#     return ax

def plot_item_ratings_freq(df):
    # item ratings overview
    rating_cnts = df['item_id'].value_counts()
    print(rating_cnts.head(10))
    print(rating_cnts.tail(10))
    plot_kwargs = {
        'kind':'bar',
        #'title':'Item ratings count',
        'figsize':(8,5),
        'xticks':range(0, rating_cnts.shape[0], 20),
        'use_index':False,
        'grid':True,
        'fontsize':14
        }
    # plot item ratings
    ax = rating_cnts.plot(**plot_kwargs)
    ax.set_xlabel('Sorted item index', fontsize = 16)
    ax.set_ylabel('Item rating count', fontsize = 16)
    return ax


def plot_user_ratings_freq(df):
    # user ratings overview
    user_cnts = df.reset_index(inplace=False)['user_id'].value_counts()
    print(user_cnts.head(10))
    print(user_cnts.tail(10))
    plot_kwargs = {
        'kind':'bar',
        #'title':'user ratings count',
        'figsize':(8,5),
        'xticks':range(0, user_cnts.shape[0], 20),
        'use_index':False,
        'grid':True,
        'fontsize':14
        }
    # plot user ratings
    ax = user_cnts.plot(**plot_kwargs)
    ax.set_xlabel('Sorted user index', fontsize = 16)
    ax.set_ylabel('User rating count', fontsize = 16)


def calc_similarity(it1, it2):
    s = it1.merge(it2, on='item_id', how='inner')
    u1_mean = it1['rating'].mean()
    u2_mean = it2['rating'].mean()
    

    s['rating_x'] -= u1_mean
    s['rating_y'] -= u2_mean
    
    similarity = sum(s['rating_x'] * s['rating_y'])
    norm = np.sqrt(sum(s['rating_x']**2) * sum(s['rating_y']**2))

    if s.empty or norm == 0:
        return 0

    return similarity/norm


def top_neighbours(u_id,df,k):
    u = df.loc[df['user_id'] == u_id] 
    # find all items rated by user
    u_its = u['item_id']
    # find all users that rated any of the items
    neighbours =  df[df['item_id'].isin(u_its)]['user_id']
    # remove the user itself
    #neighbours = neighbours[neighbours != u]
    # calculate similarity
    sims = []
    for n in neighbours:
        sim_val = calc_similarity(u,df.loc[df['user_id'] == n])
        sims.append((n,sim_val))
    sims.sort(reverse=True, key=lambda x: x[1])
    return sims[:k]









def calc_similarity_matrix(df):
    users = df['user_id'].unique()
    m = pd.DataFrame(users)
    m = m.merge(m, how='cross')
    p = np.full(m.shape[0],-1.1)
    m['similarity'] = p
    m.columns = ['user_id_x','user_id_y','similarity']
    m.set_index(['user_id_x','user_id_y'], inplace=True)


    o = 0

    for u,v in m.index:
        u_data = df.loc[df['user_id'] == u]
        if m.loc[(u,v)].values[0] == -1.1:
            if m.loc[(v,u)].values[0] == -1.1:
                v_data = df.loc[df['user_id'] == v]
                m.loc[(v,u)] = calc_similarity(v_data,u_data)
            else:
                m.loc[(u,v)] = m.loc[(v,u)]
        else:
            pass
        

        if o % 1000 == 0:
            print(o)
            
        o += 1
    return m    




def get_user(user_id,df):
    return df.loc[df['user_id'] == user_id]

def get_item(item_id,df):
    return df.loc[df['item_id'] == item_id]

def get_user_item(user_id,item_id,df):
    return df.loc[(df['user_id'] == user_id) & (df['item_id'] == item_id)]

def get_users(df,uids):
    return df.loc[df['user_id'].isin(uids)]

def get_items(df,iids):
    return df.loc[df['item_id'].isin(iids)]
