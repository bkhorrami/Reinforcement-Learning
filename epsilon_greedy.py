__author__ = 'babak_khorrami'


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def eliminate_dup(mat):
    """
    eliminates the duplicate rows of a matrix
    """
    x = np.random.rand(mat.shape[1])
    y = mat.dot(x)
    unq, index = np.unique(y, return_index=True)
    return mat[index]

def random_lot(num,crop_list,row,col,elim=np.array([])):
    """
    :param num: number of random lots
    :param crop: number of crops for which the random lot is produced
    :param row: row count in the grid
    :param col: column count in the grid
    :return:
    """
    rpt = 2
    cnt = 0
    rnd_grid = np.zeros((num,2))
    while True:
        rx=np.random.randint(0,row,rpt * num).reshape((rpt * num,))
        cx=np.random.randint(0,col,rpt * num).reshape((rpt * num,))
        tmp = np.c_[rx,cx]
        u=eliminate_dup(tmp)
        if elim.shape[0] !=0 : #if there are rows to be eliminated:
            ix=np.nonzero(np.all(u == elim[:,np.newaxis], axis=2))[1] #find the row index
            u = u[np.setdiff1d(np.arange(u.shape[0]),ix),:]
        cnt = u.shape[0]
        if cnt<num:
            rpt = rpt + 1
            continue
        else:
            np.copyto(rnd_grid,u[0:num,:])
            break

    x1 = rnd_grid[:,0]
    x2 = rnd_grid[:,1]
    np.random.shuffle(x1)
    np.random.shuffle(x2)
    rnd_grid = np.c_[x1,x2]
    #** Crop index
    rem = num%len(crop_list)
    crp_idx = []
    if rem == 0:
        for c in crop_list:
            crp_idx = crp_idx + [c] * int(num/len(crop_list))
        crp_idx = np.array(crp_idx).reshape((num,1))
    else:
        count = num//len(crop_list)
        for c in crop_list:
            crp_idx = crp_idx + [c] * count
        rem_crop = crop_list[0:rem]
        crp_idx = crp_idx + rem_crop

    crp_idx = np.array(crp_idx).reshape((len(crp_idx)))
    result = np.c_[crp_idx,rnd_grid]
    return result


#*** Function to simulate the yield potentials of different lots for each crop and season:
def crop_yield(seas,crop,row,col):
    if seas>3 or seas<0 or crop>=20 or crop<0 or row>=50 or row<0 or col>=20 or col<0:
        print("Input is incorrect")
        return

    if seas == 0:
        if crop >=0 and crop < 5:
            if row >= 0 and row <=20 and col>=0 and col <= 10:
                yld = np.random.normal(70,3)
            else:
                yld = np.random.normal(20,5)
        else:
            yld = np.random.normal(30,5)
    elif seas == 1:
        if crop >=5 and crop < 10:
            if row >= 20 and row <= 30 and col >= 10 and col <= 20:
                yld = np.random.normal(90,3)
            else:
                yld = np.random.normal(20,5)
        else:
            yld = np.random.normal(30,5)
    elif seas == 2:
        if crop >=10 and crop < 15:
            if row >= 30 and row <= 40 and col >= 15 and col <= 20:
                yld = np.random.normal(60,3)
            else:
                yld = np.random.normal(25,5)
        else:
            yld = np.random.normal(30,5)
    elif seas ==3:
        if crop >=15 and crop <=19:
            if row >= 40 and row <= 50 and col>=0 and col <= 10:
                yld = np.random.normal(60,3)
            else:
                yld = np.random.normal(25,5)
        else:
            yld = np.random.normal(30,5)
    else:
        yld = np.random.normal(20,5)

    return yld

def optimize_reward(crop_list,row,col,time_period,eps):
    sell_price = list(np.random.uniform(20,40,20)) #randomly generated selling prices
    cost = 10
    expr_count = [250,250,250,250] # No. of trials for seasons 1..4
    q=np.zeros((4,20,50,20)) #initialize (season , crop , row , col)
    q_counts=np.zeros((4,20,50,20)) #count initialize (season , crop , row , col)
    annual_profit = []
    seasonal_profit = []
    
    # Running the experiments for the first period (year)
    year = 1 # first year
    for season in range(0,4): #loop over seasons for the first year.
        seas_reward , yr_reward = 0 , 0
        lots = random_lot(expr_count[season],crop_list,row,col)
        for i in range(lots.shape[0]):
            crp,r,c = int(lots[i,0]),int(lots[i,1]),int(lots[i,2])
            yld = crop_yield(season,crp,r,c)
            reward = yld*sell_price[crp]
            seas_reward+=reward
            q_counts[season,crp,r,c] += 1
            q[season,crp,r,c] = q[season,crp,r,c] +\
                                    (1/q_counts[season,crp,r,c]) * (reward - q[season,crp,r,c])
        seasonal_profit.append(seas_reward)
        yr_reward+=seas_reward

    annual_profit.append(yr_reward)



    #*** For years 2..10:
    for year in range(2,time_period): #time_period + 1
        yr_reward = 0
        for season in range(0,4):
            seas_reward = 0
            top_crop = int((1-eps) * expr_count[season]) #repeat the top performer crops/lots for the season
            max_idx_tmp=np.dstack(np.unravel_index(np.argsort(-q[season,:,:,:].ravel()), (20,50,20))) #sort
            max_idx = max_idx_tmp[0,:,:]
            crp_id,r,c = max_idx[0:top_crop,0],max_idx[0:top_crop,1],max_idx[0:top_crop,2]
            top_lots = np.c_[crp_id,np.c_[r,c]]
            explore_count = expr_count[season] - len(crp_id) #How many new lots need to be explored
            explore_lots = random_lot(explore_count,crop_list,row,col,np.c_[r,c])
            expr_lots = np.vstack((top_lots,explore_lots))

            #*** Explore and Exploit
            for i in range(expr_lots.shape[0]):
                crp,r,c = int(expr_lots[i,0]),int(expr_lots[i,1]),int(expr_lots[i,2])
                yld = crop_yield(season,crp,r,c)
                reward = yld*sell_price[crp]
                seas_reward += reward
                q_counts[season,crp,r,c] += 1
                q[season,crp,r,c] = q[season,crp,r,c] +\
                                    (1/q_counts[season,crp,r,c])*(reward - q[season,crp,r,c])
        eps = eps*0.99
        seasonal_profit.append(seas_reward)
        yr_reward+=seas_reward
        annual_profit.append(yr_reward)

    return annual_profit

def main():
    annual_profit_3 = optimize_reward(list(range(20)),50,20,500,0.3)
    annual_profit_2 = optimize_reward(list(range(20)),50,20,500,0.2)


    # print(annual_profit_3)
    plt.plot(annual_profit_3,'-')
    plt.plot(annual_profit_2,'-')

    plt.xlabel('epochs')
    plt.ylabel('Profit')
    plt.show()


if __name__ == '__main__':
    main()



