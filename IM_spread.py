from utils import *



def IM_spread(dataset_name,file_name,seed_size):


    data_path = './data/%s.txt' %(dataset_name)  

    # obtain the set of seed nodes of PrivGraph
    S = find_seed(file_name,seed_size=seed_size)

    # calculate the influence spread
    influence_spread = cal_spread(data_path,S_all=S,seed_size=seed_size)

    return influence_spread


if __name__ == '__main__':
    epsilon = 1.5

    seed_size = 20

    # set the dataset
    # dataset_name = 'Enron'
    # dataset_name = 'CA-HepPh'
    # dataset_name = 'Facebook'
    dataset_name = 'Chamelon'

    root_path = './result/'

    # import the txt file
    file_name = root_path + 'PrivGraph_%s_%.1f.txt'  %(dataset_name,epsilon)

    print('dataset:%s,epsilon:%.1f,seed_size:%d'%(dataset_name,epsilon,seed_size))

    influence_spread = IM_spread(dataset_name,file_name,seed_size)

    print('Influence Spread:',influence_spread)

