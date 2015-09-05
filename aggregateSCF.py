import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

def scf_aggregate(year, filepath):

    df = pd.read_stata(filepath)

    CPI_U_RS = {1983:155.6, 1989:190.2, 1992:211.6, 1995:226.5, 1998:240.5,
                2001:261.8, 2004:278.8, 2007:306.2, 2010:320.8, 2013:343.8}

    def deflate(CPI_dict, baseyear = 2013):
        for year in CPI_dict:
            if year != baseyear:
                CPI_dict[year] = CPI_dict[year]/CPI_dict[baseyear]

    CPI_deflator = deflate(CPI_U_RS, baseyear = year)

    def wmean(series):
        return sum(series*df.wgt/sum(df.wgt))

    if year == 1989:
        df.rename(columns = {'XX1':'IID'}, inplace=True)
    else:
        df.rename(columns = {'YY1':'IID'}, inplace=True)

    # only keep observations with valid ID and weight
    cond = (df.X42001 > 0) & (df.IID > 0)
    if len(df[~cond])>0:
        print "Dropped",len(df[~cond]),"observations without valid weight or ID"
    df = df[(df.X42001 > 0) & (df.IID > 0)]

    # Divide weight by 5 so totals estimated on the 5 implicates jointly are correct
    df['wgt'] = df.X42001/5

    # Retain original weight: df.wgt0
    df['wgt0'] = df.X42001

    df.rename(columns = {'X8021': 'hhsex', 'X14':'age', 'X5901':'educ', }, inplace=True)

    df['agecl'] = (1 + (df.age>34).astype(int) + (df.age>44).astype(int) + (df.age>54).astype(int)
                     + (df.age>64).astype(int) + (df.age>74).astype(int) )

    df['edcl'] = 1
    df.loc[df.X5904==1, 'edcl'] = 4
    df.loc[df.educ>12, 'edcl'] = 3
    df.loc[df.X5902.isin([1,2]), 'edlc'] = 2#

    df['married'] = 2
    df.loc[df.X8023.isin([1,2]), 'married'] = 1

    kidvars = ['X108', 'X114', 'X120', 'X126', 'X132', 'X202', 'X208', 'X214', 'X220', 'X226']
    df['kids'] = 0
    if year < 1995:
        for var in kidvars:
            df['kids'] += df[var].isin([4,13,36]).astype(int)
    else:
        for var in kidvars[:-1]:
            df['kids'] += df[var].isin([4,13,36]).astype(int)

    df['lf'] = 1
    df.loc[np.logical_or((df.X4100>=50)&(df.X4100<=80), df.X4100==97), 'lf'] = 0

    df['lifecl'] = 0
    df.loc[(df.age<55)&(df.married!=1)&(df.kids)==0, 'lifecl'] = 1
    df.loc[(df.age<55)&(df.married==1)&(df.kids)==0, 'lifecl'] = 2
    df.loc[(df.age<55)&(df.married==1)&(df.kids) >0, 'lifecl'] = 3
    df.loc[(df.age<55)&(df.married!=1)&(df.kids) >0, 'lifecl'] = 4
    df.loc[(df.age>54)&(df.lf==1), 'lifecl'] = 5
    df.loc[(df.age>54)&(df.lf==2), 'lifecl'] = 5

    df['famstruct'] = 0
    df.loc[(df.married!=1)&(df.kids>0), 'famstruct'] = 1
    df.loc[(df.married!=1)&(df.kids==0)&(df.age<55), 'famstruct'] = 2
    df.loc[(df.married!=1)&(df.kids==0)&(df.age>54), 'famstruct'] = 3
    df.loc[(df.married==1)&(df.kids>0), 'famstruct'] = 4
    df.loc[(df.married==1)&(df.kids==0), 'famstruct'] = 5
    if len(df[df.famstruct==0]) > 0:
        print "There are",len(df[df.famstruct==0]),"observations with unclassified family structure"

    df['occat1'] = 0
    df.loc[df.age<65, 'occat1'] = 4
    df.loc[df.X4106==1, 'occat1'] = 1
    df.loc[df.X4106.isin([2,3,4]), 'occat1'] = 2
    df.loc[np.logical_or(df.X4100.isin([50,52]), (df.X4100.isin([21,23,30,70,80,97,85,-7]))&(df.age>64)), 'occat1'] = 3

    df['occat2'] = 0
    df.loc[df.X7401==1, 'occat2'] = 1
    df.loc[df.X7401.isin([2,3]),'occat2'] = 2
    df.loc[df.X7401.isin([4,5,6]),'occat2'] = 3
    df.loc[df.X7401==0, 'occat2'] = 4

    df['indcat'] = 2
    df.loc[df.occat1>2, 'indcat'] = 4
    df.loc[df.X7402.isin([2,3,]), 'indcat'] = 1

    if year == 1989:
        df.rename(columns = {'X40083':'region'}, inplace=True)
    else:
        df.rename(columns = {'X30022':'region'}, inplace=True)

    df['income'] = df.X5729.clip(0,np.inf)
    df['wageinc'] = df.X5702
    df['businessfarminc'] = df.X5704 + df.X5714
    df['intdvinc'] = df.X5706 + df.X5708 + df.X5710
    df['kginc'] = df.X5712
    df['ssretinc'] = df.X5722
    df['transfothinc'] = df.X5716 + df.X5718 + df.X5720 + df.X5724
    df['rentinc'] = df.X5714

    if (year < 2010) & (year!=2004):
        df['grossinc_taxsim'] = df[['pwages', 'dividends', 'pensions', 'gssi', 'otherprop', 'stcg', 'ltcg']].sum(axis=1)
        df['avtax_rate'] = df[['fiitax', 'siitax', 'fica']].sum(axis=1)/df.grossinc_taxsim
        df.avtax_rate = df['avtax_rate'].clip(0,1)

    df['checking'] =(df.X3506.clip(0,np.inf)*(df.X3507==5) + df.X3510.clip(0,np.inf)*(df.X3511==5)
                   + df.X3514.clip(0,np.inf)*(df.X3515==5) + df.X3518.clip(0,np.inf)*(df.X3519==5)
                   + df.X3522.clip(0,np.inf)*(df.X3522==5) + df.X3526.clip(0,np.inf)*(df.X3527==5)
                   + df.X3529.clip(0,np.inf)*(df.X3507==5) )

    if year <= 2001:
        # for 1989 - 2001
        df['saving'] =(df.X3804.clip(0,np.inf) + df.X3807.clip(0,np.inf) + df.X3810.clip(0,np.inf) + df.X3813.clip(0,np.inf)
                      + df.X3816.clip(0,np.inf) + df.X3818.clip(0,np.inf) )
    else:
        # for 2004 - 2010
        df['saving'] = (df.X3730.clip(0,np.inf)*(~df.X3732.isin([4,30])) + df.X3736.clip(0,np.inf)*(~df.X3738.isin([4,30]))
                      +  df.X3742.clip(0,np.inf)*(~df.X3744.isin([4,30])) + df.X3748.clip(0,np.inf)*(~df.X3750.isin([4,30]))
                      +  df.X3754.clip(0,np.inf)*(~df.X3756.isin([4,30])) + df.X3760.clip(0,np.inf)*(~df.X3762.isin([4,30]))
                      +  df.X3765.clip(0,np.inf) )

    if year <= 2001:
        df['mm_dep'] = (df.X3506.clip(0,np.inf)*((df.X3507==1).astype(int)*(df.X9113.isin([11,12,13])))
                     + df.X3510.clip(0,np.inf)*((df.X3511==1).astype(int)*(df.X9114.isin([11,12,13])))
                     + df.X3514.clip(0,np.inf)*((df.X3515==1).astype(int)*(df.X9115.isin([11,12,13])))
                     + df.X3518.clip(0,np.inf)*((df.X3519==1).astype(int)*(df.X9116.isin([11,12,13])))
                     + df.X3522.clip(0,np.inf)*((df.X3523==1).astype(int)*(df.X9117.isin([11,12,13])))
                     + df.X3526.clip(0,np.inf)*((df.X3527==1).astype(int)*(df.X9118.isin([11,12,13])))
                     + df.X3529.clip(0,np.inf)*((df.X3527==1).astype(int)*(df.X9118.isin([11,12,13])))
                     + df.X3706.clip(0,np.inf)*(df.X9131.isin([11,12,13])) + df.X3711.clip(0,np.inf)*(df.X9132.isin([11,12,13]))
                     + df.X3716.clip(0,np.inf)*(df.X9133.isin([11,12,13])) + df.X3718.clip(0,np.inf)*(df.X9133.isin([11,12,13])))
    else:
        df['mm_dep'] = (df.X3506.clip(0,np.inf)*((df.X3507==1).astype(int)*(df.X9113.isin([11,12,13])))
                     + df.X3510.clip(0,np.inf)*((df.X3511==1).astype(int)*(df.X9114.isin([11,12,13])))
                     + df.X3514.clip(0,np.inf)*((df.X3515==1).astype(int)*(df.X9115.isin([11,12,13])))
                     + df.X3518.clip(0,np.inf)*((df.X3519==1).astype(int)*(df.X9116.isin([11,12,13])))
                     + df.X3522.clip(0,np.inf)*((df.X3523==1).astype(int)*(df.X9117.isin([11,12,13])))
                     + df.X3526.clip(0,np.inf)*((df.X3527==1).astype(int)*(df.X9118.isin([11,12,13])))
                     + df.X3529.clip(0,np.inf)*((df.X3527==1).astype(int)*(df.X9118.isin([11,12,13])))
                     + df.X3730.clip(0,np.inf)*(df.X3732.isin([4,30]).astype(int))*(df.X9259.isin([11,12,13])).astype(int)
                     + df.X3736.clip(0,np.inf)*(df.X3738.isin([4,30]).astype(int))*(df.X9260.isin([11,12,13])).astype(int)
                     + df.X3742.clip(0,np.inf)*(df.X3744.isin([4,30]).astype(int))*(df.X9261.isin([11,12,13])).astype(int)
                     + df.X3748.clip(0,np.inf)*(df.X3750.isin([4,30]).astype(int))*(df.X9262.isin([11,12,13])).astype(int)
                     + df.X3754.clip(0,np.inf)*(df.X3756.isin([4,30]).astype(int))*(df.X9263.isin([11,12,13])).astype(int)
                     + df.X3760.clip(0,np.inf)*(df.X3762.isin([4,30]).astype(int))*(df.X9264.isin([11,12,13])).astype(int)
                     + df.X3765.clip(0,np.inf)*(df.X3762.isin([4,30]).astype(int))*(df.X9264.isin([11,12,13])).astype(int))


    if year <= 2001:
        df['mmmf'] = (df.X3506.clip(0,np.inf)*((df.X3507==1) & (~df.X9113.isin([11,12,13]))).astype(int)
                    + df.X3510.clip(0,np.inf)*((df.X3511==1) & (~df.X9114.isin([11,12,13]))).astype(int)
                    + df.X3514.clip(0,np.inf)*((df.X3515==1) & (~df.X9115.isin([11,12,13]))).astype(int)
                    + df.X3518.clip(0,np.inf)*((df.X3519==1) & (~df.X9116.isin([11,12,13]))).astype(int)
                    + df.X3522.clip(0,np.inf)*((df.X3523==1) & (~df.X9117.isin([11,12,13]))).astype(int)
                    + df.X3526.clip(0,np.inf)*((df.X3527==1) & (~df.X9118.isin([11,12,13]))).astype(int)
                    + df.X3529.clip(0,np.inf)*((df.X3527==1) & (~df.X9118.isin([11,12,13]))).astype(int)
                    + df.X3706.clip(0,np.inf)*(~df.X9131.isin([11,12,13])).astype(int)
                    + df.X3711.clip(0,np.inf)*(~df.X9132.isin([11,12,13])).astype(int)
                    + df.X3716.clip(0,np.inf)*(~df.X9133.isin([11,12,13])).astype(int)
                    + df.X3718.clip(0,np.inf)*(~df.X9133.isin([11,12,13])).astype(int) )
    else:
        df['mmmf'] = (df.X3506.clip(0,np.inf)*((df.X3507==1) & (~df.X9113.isin([11,12,13]))).astype(int)
                    + df.X3510.clip(0,np.inf)*((df.X3511==1) & (~df.X9114.isin([11,12,13]))).astype(int)
                    + df.X3514.clip(0,np.inf)*((df.X3515==1) & (~df.X9115.isin([11,12,13]))).astype(int)
                    + df.X3518.clip(0,np.inf)*((df.X3519==1) & (~df.X9116.isin([11,12,13]))).astype(int)
                    + df.X3522.clip(0,np.inf)*((df.X3523==1) & (~df.X9117.isin([11,12,13]))).astype(int)
                    + df.X3526.clip(0,np.inf)*((df.X3527==1) & (~df.X9118.isin([11,12,13]))).astype(int)
                    + df.X3529.clip(0,np.inf)*((df.X3527==1) & (~df.X9118.isin([11,12,13]))).astype(int)
                    + df.X3730.clip(0,np.inf)*((df.X3732.isin([4,30])) & (~df.X9259.isin([11,12,13]))).astype(int)
                    + df.X3736.clip(0,np.inf)*((df.X3738.isin([4,30])) & (~df.X9260.isin([11,12,13]))).astype(int)
                    + df.X3742.clip(0,np.inf)*((df.X3744.isin([4,30])) & (~df.X9261.isin([11,12,13]))).astype(int)
                    + df.X3748.clip(0,np.inf)*((df.X3750.isin([4,30])) & (~df.X9262.isin([11,12,13]))).astype(int)
                    + df.X3754.clip(0,np.inf)*((df.X3756.isin([4,30])) & (~df.X9263.isin([11,12,13]))).astype(int)
                    + df.X3760.clip(0,np.inf)*((df.X3762.isin([4,30])) & (~df.X9264.isin([11,12,13]))).astype(int)
                    + df.X3765.clip(0,np.inf)*((df.X3762.isin([4,30])) & (~df.X9264.isin([11,12,13]))).astype(int) )

    df['mma'] = df.mm_dep + df.mmmf

    df['call'] = df.X3930.clip(0,np.inf)

    df['liquid'] = df[['checking', 'saving', 'mma', 'call']].sum(axis=1)

    df['cds'] = df.X3721.clip(0,np.inf)

    df['stmutf'] = df.X3822.clip(0,np.inf)*(df.X3821==1)
    df['tfbmutf'] = df.X3824.clip(0,np.inf)*(df.X3823==1)
    df['gbmutf'] = df.X3826.clip(0,np.inf)*(df.X3825==1)
    df['obmutf'] = df.X3828.clip(0,np.inf)*(df.X3827==1)
    df['comutf'] = df.X3830.clip(0,np.inf)*(df.X3829==1)

    if year > 2004:
        df['omutf'] = df.X7787.clip(0,np.inf)*(df.X7785==1)
    else:
        df['omutf'] = 0

    df['nmmf'] = df[['stmutf', 'tfbmutf', 'gbmutf', 'obmutf', 'comutf', 'omutf']].sum(axis=1)

    df['mortbnd'] = df.X3906
    df['govtbnd'] = df.X3908
    df['notxbnd'] = df.X3910

    if year >= 1992:
        df['obnd'] = df.X7634 + df.X7633
    else:
        df['obnd'] = df.X3912

    df['bonds'] = df[['notxbnd', 'mortbnd', 'govtbnd', 'obnd']].sum(axis=1)

    df['savbnd'] = df.X3902

    df['stocks'] = df.X3915.clip(0,np.inf)

    if year >= 2004:
        df['annuit'] = df.X6577.clip(0,np.inf)
        df['trusts'] = df.X6587.clip(0,np.inf)
        df['othma'] = df.annuit + df.trusts
    elif year in [1998,2001]:
        df['annuit'] = df.X6820.clip(0,np.inf)
        df['trusts'] = df.X6835.clip(0,np.inf)
        df['othma'] = df.annuit + df.trusts
    else:
        df['othma'] = df.X3942.clip(0,np.inf)
        denom = ((df.X3934==1).astype(int) + (df.X3935==1).astype(int)
               + (df.X3936==1).astype(int) + (df.X3937==1).astype(int)).clip(1,np.inf)
        df['annuit'] = ( (df.X3935==1).astype(int)/ denom )*df.X3942.clip(0,np.inf)
        df['trusts'] = df['othma'] - df['annuit']

    df['cashli'] = df.X4006.clip(0,np.inf)

    df['othfin'] = (df.X4018
                   + df.X4022*(df.X4020.isin([61,62,63,64,65,66,67,72,73,74,77,79,80,81,82,83,84]))
                   + df.X4026*(df.X4024.isin([61,62,63,64,65,66,67,72,73,74,77,79,80,81,82,83,84]))
                   + df.X4030*(df.X4028.isin([61,62,63,64,65,66,67,72,73,74,77,79,80,81,82,83,84])) )

    if year >= 2004:
        df['irakh'] = df[['X6551','X6559','X6567','X6552','X6560','X6568','X6553','X6561',
                          'X6569','X6554','X6562','X6570']].sum(axis=1)
    else:
        df['irakh'] = df.X3610.clip(0,np.inf) + df.X3620.clip(0,np.inf) + df.X3630.clip(0,np.inf)

    if year < 2004:
        ptype = ['X4216', 'X4316', 'X4416', 'X4816', 'X4916', 'X5016']
        pamt  = ['X4226', 'X4326', 'X4426', 'X4826', 'X4926', 'X5026']
        pbor  = ['X4227', 'X4327', 'X4427', 'X4827', 'X4927', 'X5027']
        pwit  = ['X4231', 'X4331', 'X4431', 'X4831', 'X4931', 'X5031']
        pall  = ['X4234', 'X4334', 'X4434', 'X4834', 'X4934', 'X5034']
        df['thrift'] = 0; df['peneq'] = 0; df['rthrift'] = 0; df['sthrift'] = 0; df['req'] = 0; df['seq'] = 0
        for i in range(0,len(ptype)):
            df['hold'] = df[pamt[i]].clip(0,np.inf)*((df[ptype[i]].isin([1,2,7,11,12,18]))
                                                   | (df[pbor[i]]==1) | (df[pwit[i]]==1)).astype(int)

            if i < 3:
                df.rthrift += df.hold
            else:
                df.sthrift += df.hold

            df.thrift += df.hold
            df.peneq += df.hold*((df[pall[i]]==1).astype(int) + 0.5*(df[pall[i]]==3).astype(int))

            if i < 3:
                df.req = df.peneq
            else:
                df.seq = df.peneq - df.req

        varlist = [1,2,7,11,12,18]
        df['pmop'] = np.nan
        df.loc[(df.X4436>0) & (df.X4216.isin(varlist) | (df.X4316.isin(varlist)) | df.X4416.isin(varlist) | (df.X4231==1)
                | (df.X4331==1) | (df.X4431==1) | (df.X4227==1) | (df.X4327==1) | (df.X4427==1)), 'pmop'] = df.X4436
        df.loc[(df.X4436>0) & (df.X4216!=0)&(df.X4316!=0)&(df.X4416!=0)&(df.X4231!=0)&(df.X4331!=0)&(df.X4431!=0),'pmop'] = 0
        df.loc[(df.X4436>0) & (np.isnan(df.pmop)), 'pmop'] = df.X4436

        df.thrift += df.pmop

        df.loc[df.req>0, 'peneq'] += df.pmop*(df.req/df.rthrift)
        df.loc[df.req<=0, 'peneq'] += df.pmop/2

        df['pmop'] = np.nan
        df.loc[(df.X5036>0) & (df.X4816.isin(varlist) | (df.X4916.isin(varlist)) | df.X5016.isin(varlist) | (df.X4831==1)
                | (df.X4931==1) | (df.X5031==1) | (df.X4827==1) | (df.X4927==1) | (df.X5027==1)), 'pmop'] = df.X5036
        df.loc[(df.X5036>0) & (df.X4816!=0)&(df.X4916!=0)&(df.X5016!=0)&(df.X4831!=0)&(df.X4931!=0)&(df.X5031!=0),'pmop'] = 0
        df.loc[(df.X5036>0) & (np.isnan(df.pmop)), 'pmop'] = df.X5036

        df.thrift += df.pmop

        df.loc[df.seq>0, 'peneq'] += df.pmop*(df.seq/df.sthrift)
        df.loc[df.seq<=0, 'peneq'] += df.pmop/2

    elif (year >= 2004) & (year<2010):
        ptype1 = ['X11000', 'X11100', 'X11200', 'X11300', 'X11400', 'X11500']
        ptype2 = ['X11001', 'X11101', 'X11201', 'X11301', 'X11401', 'X11501']
        pamt = ['X11032', 'X11132', 'X11232', 'X11332', 'X11432', 'X11532']
        pbor = ['X11025', 'X11125', 'X11225', 'X11325', 'X11425', 'X11525']
        pwit = ['X11031', 'X11131', 'X11231', 'X11331', 'X11431', 'X11531']
        pall = ['X11036', 'X11136', 'X11236', 'X11336', 'X11436', 'X11536']
        ppct = ['X11037', 'X11137', 'X11237', 'X11337', 'X11437', 'X11537']
        df['thrift'] = 0; df['peneq'] = 0; df['rthrift'] = 0; df['sthrift'] = 0; df['req'] = 0; df['seq'] = 0

        for i in range(0,len(ptype1)):
            df['hold'] = (df[pamt[i]].clip(0,np.inf)*( (df[ptype1[i]].isin([5,6,10,21]))
                                                     | (df[ptype2[i]].isin([2,3,4,6,20,21,22,26]))
                                                     | (df[pbor[i]]==1)
                                                     | (df[pwit[i]]==1)).astype(int) )

            if i < 3:
                df.rthrift += df.hold
            else:
                df.sthrift += df.hold

            df.thrift += df.hold
            df.peneq += df.hold*((df[pall[i]]==1).astype(int) + (df[ppct[i]].clip(0,np.inf))*(df[pall[i]]==3).astype(int)/10000)

            if i < 3:
                df.req = df.peneq
            else:
                df.seq = df.peneq - df.req

        df['hold'] = np.nan; df['pmop'] = np.nan
        varlist1 = [5,6,10,21]
        varlist2 = [2,3,4,6,20,21,22,26]
        df.loc[(df.X11259>0) & (df.X11000.isin(varlist1) | (df.X11100.isin(varlist1)) | df.X11200.isin(varlist1)
                             | (df.X11001.isin(varlist2) | (df.X11101.isin(varlist2)) | df.X11200.isin(varlist2)
                             | (df.X11031==1) | (df.X11131==1) | (df.X11231==1)
                             | (df.X11025==1) | (df.X11125==1))| (df.X11225==1)), 'pmop'] = df.X11259
        df.loc[(df.X11259>0) & (df.X11000!=0)&(df.X11100!=0)&(df.X11200!=0)&(df.X11025!=0)&(df.X11125!=0)&(df.X11225!=0),'pmop'] = 0
        df.loc[(df.X11259>0) & (np.isnan(df.pmop)), 'pmop'] = df.X11259

        df.thrift += df.pmop

        df.loc[df.req>0, 'peneq'] += df.pmop*(df.req/df.rthrift)
        df.loc[df.req<=0, 'peneq'] += df.pmop/2

        df.loc[(df.X11559>0) & (df.X11300.isin(varlist1) | (df.X11400.isin(varlist1)) | df.X11500.isin(varlist1)
                             | (df.X11301.isin(varlist2) | (df.X11401.isin(varlist2)) | df.X11500.isin(varlist2)
                             | (df.X11331==1) | (df.X11431==1) | (df.X11531==1)
                             | (df.X11325==1) | (df.X11425==1))| (df.X11525==1)), 'pmop'] = df.X11559
        df.loc[(df.X11559>0) & (df.X11300!=0)&(df.X11400!=0)&(df.X11500!=0)&(df.X11325!=0)&(df.X11425!=0)&(df.X11525!=0),'pmop'] = 0
        df.loc[(df.X11559>0) & (np.isnan(df.pmop)), 'pmop'] = df.X11559

        df.loc[df.seq>0, 'peneq'] += df.pmop*(df.seq/df.sthrift)
        df.loc[df.seq<=0, 'peneq'] += df.pmop/2

    else:
        ptype1 = ['X11000', 'X11100', 'X11300', 'X11400']
        ptype2 = ['X11001', 'X11101', 'X11301', 'X11401']
        pamt =   ['X11032', 'X11132', 'X11332', 'X11432']
        pbor =   ['X11025', 'X11125', 'X11325', 'X11425']
        pwit =   ['X11031', 'X11131', 'X11331', 'X11431']
        pall =   ['X11036', 'X11136', 'X11336', 'X11436']
        ppct =   ['X11037', 'X11137', 'X11337', 'X11437']
        df['thrift'] = 0; df['peneq'] = 0; df['rthrift'] = 0; df['sthrift'] = 0; df['req'] = 0; df['seq'] = 0
        varlist = [2,3,4,6,20,21,22,26]

        for i in range(0,len(ptype1)):
            df['hold'] = df[pamt[i]].clip(0,np.inf)*((df[ptype1[i]]==1) | (df[ptype2[i]].isin(varlist))
                                                     | (df[pbor[i]]==1) | (df[pwit[i]]==1) )
            if i < 2:
                df.rthrift += df.hold
            else:
                df.sthrift += df.hold

            df.thrift += df.hold
            df['peneq'] += df.hold*((df[pall[i]]==1).astype(int) + (df[ppct[i]].clip(0,np.inf))*(df[pall[i]].isin([3,30])).astype(int)/10000)

            if i < 2:
                df.req = df.peneq
            else:
                df.seq = df.peneq - df.req

        df['hold'] = np.nan; df['pmop'] = np.nan
        varlist = [2,3,4,6,20,21,22,26]
        df.loc[(df.X11259>0) & ((df.X11000==1) | (df.X11100==1) | (df.X11001.isin(varlist) | (df.X11101.isin(varlist))
                               |(df.X11031==1) | (df.X11131==1) | (df.X11025==1) | (df.X11125==1))), 'pmop'] = df.X11259
        df.loc[(df.X11259>0) & (df.X11000!=0) & (df.X11100!=0) & (df.X11025!=0) & (df.X11125!=0),'pmop'] = 0
        df.loc[(df.X11259>0) & (np.isnan(df.pmop)), 'pmop'] = df.X11259

        df.thrift += df.pmop

        df.loc[df.req>0, 'peneq'] += df.pmop*(df.req/df.rthrift)
        df.loc[df.req<=0, 'peneq'] += df.pmop/2

        df.loc[(df.X11559>0) & ((df.X11300==1) | (df.X11400==1) | (df.X11301.isin(varlist)) | (df.X11401.isin(varlist))
                             | (df.X11331==1) | (df.X11431==1) | (df.X11325==1) | (df.X11425==1)), 'pmop'] = df.X11559
        df.loc[(df.X11559>0) & (df.X11300!=0) & (df.X11400!=0) & (df.X11325!=0) & (df.X11425!=0),'pmop'] = 0
        df.loc[(df.X11559>0) & (np.isnan(df.pmop)), 'pmop'] = df.X11559

        df.loc[df.seq>0, 'peneq'] += df.pmop*(df.seq/df.sthrift)
        df.loc[df.seq<=0, 'peneq'] += df.pmop/2

    if year >= 2010:
        df['futpen'] = df.X5604.clip(0,np.inf) + df.X5612.clip(0,np.inf) + df.X5620.clip(0,np.inf) + df.X5628.clip(0,np.inf)
    else:
        df['futpen'] = (df.X5604.clip(0,np.inf) + df.X5612.clip(0,np.inf) + df.X5620.clip(0,np.inf) + df.X5628.clip(0,np.inf)
                     +  df.X5636.clip(0,np.inf)+df.X5644.clip(0,np.inf) )

    if (year >= 2004) & (year <= 2007):
        df['currpen'] = df.X6462 + df.X6467 + df.X6472 + df.X6477 + df.X6482 + df.X6487 + df.X6957
        df['retqliq'] = df.irakh + df.thrift + df.futpen + df.currpen
    elif year >= 2010:
        df['currpen'] = df.X6462 + df.X6467 + df.X6472 + df.X6477 + df.X6957
        df['retqliq'] = df.irakh + df.thrift + df.futpen + df.currpen
    elif year == 2001:
        df['currpen'] = df.X6462 + df.X6467 + df.X6472 + df.X6477 + df.X6482 + df.X6487
        df['retqliq'] = df.irakh + df.thrift + df.futpen + df.currpen
    else:
        df['currpen'] =  0
        df['retqliq'] = df.irakh + df.thrift + df.futpen

    df['fin'] = df[['liquid', 'cds', 'nmmf', 'stocks', 'bonds', 'savbnd', 'cashli', 'othma', 'othfin']].sum(axis=1)

    if year >= 1995:
        df['vehic'] = (df.X8166.clip(0,np.inf) + df.X8167.clip(0,np.inf)+df.X8168.clip(0,np.inf) + df.X8188.clip(0,np.inf)
                     + df.X2422.clip(0,np.inf) + df.X2506.clip(0,np.inf)+df.X2606.clip(0,np.inf) + df.X2623.clip(0,np.inf) )
    else:
        df['vehic'] = (df.X8166.clip(0,np.inf) + df.X8167.clip(0,np.inf) + df.X8168.clip(0,np.inf)
                     + df.X2422.clip(0,np.inf) + df.X2506.clip(0,np.inf)+df.X2606.clip(0,np.inf) + df.X2623.clip(0,np.inf) )

    # cap percent of famr used for farming at 90%:
    df.x507 = df.X507.clip(0,9000)
    df['farmbus'] = 0

    df.farmbus = (df.X507/10000)*(df.X513 + df.X526 - df.X805 - df.X905 - df.X1005)
    df.X805 = df.X805*((10000 - df.X507)/10000)
    df.X808 = df.X808*((10000 - df.X507)/10000)
    df.X813 = df.X813*((10000 - df.X507)/10000)
    df.X905 = df.X905*((10000 - df.X507)/10000)
    df.X908 = df.X908*((10000 - df.X507)/10000)
    df.X913 = df.X913*((10000 - df.X507)/10000)
    df.X1005 = df.X1005*((10000 - df.X507)/10000)
    df.X1008 = df.X1008*((10000 - df.X507)/10000)
    df.X1013 = df.X1013*((10000 - df.X507)/10000)

    df.loc[df.X1103==1, 'farmbus'] = df.farmbus - df.X1108*(df.X507/10000)
    df.loc[df.X1103==1, 'X1108'] = df.X1108*(10000-df.X507)/10000
    df.loc[df.X1103==1, 'X1109'] = df.X1109*(10000-df.X507)/10000

    df.loc[df.X1114==1, 'farmbus'] = df.farmbus - df.X1119*(df.X507/10000)
    df.loc[df.X1114==1, 'X1119'] = df.X1119*(10000-df.X507)/10000
    df.loc[df.X1114==1, 'X1120'] = df.X1120*(10000-df.X507)/10000

    df.loc[df.X1125==1, 'farmbus'] = df.farmbus - df.X1130*(df.X507/10000)
    df.loc[df.X1125==1, 'X1130'] = df.X1130*(10000-df.X507)/10000
    df.loc[df.X1125==1, 'X1131'] = df.X1131*(10000-df.X507)/10000

    cond = (df.X1136>0) & (df.X1108 + df.X1119 + df.X1130 > 0)

    df.loc[cond,'farmbus'] = (df.farmbus - df.X1136*(df.X507/10000)*(df.X1108*(df.X1103==1).astype(int)
                           + df.X1119*(df.X1114==1).astype(int)
                           + df.X1130*(df.X1125==1).astype(int)) /(df.X1108 + df.X1119 + df.X1130))

    df.loc[cond,'X1136'] = (df.X1136*((10000-df.X507)/10000)*(df.X1108*(df.X1103==1).astype(int)
                         + df.X1119*(df.X1114==1).astype(int)
                         + df.X1130*(df.X1125==1).astype(int)) /(df.X1108 + df.X1119 + df.X1130))

    df['houses'] = df[['X604', 'X614', 'X623', 'X716']].sum(axis=1) + ((10000-df.X507.clip(0,np.inf))/10000)*(df.X513 + df.X526)


    if year >= 2013:
        df['oresre'] = (df[['X1306','X1310']].max(axis=1) + df[['X1325','X1329']].max(axis=1) + df.X1339.clip(0,np.inf)
                      + df.X1706.clip(0,np.inf)*(df.X1705/10000)*(df.X1703.isin([12,14,21,22,25,40,41,42,43,44,49,50,52,999])).astype(int)
                      + df.X1806.clip(0,np.inf)*(df.X1805/10000)*(df.X1803.isin([12,14,21,22,25,40,41,42,43,44,49,50,52,999])).astype(int)
                      + df.X2002.clip(0,np.inf) )
    elif year==2010:
        df['oresre'] = (df[['X1405','X1409']].max(axis=1) + df[['X1505','X1509']].max(axis=1) + df.X1619.clip(0,np.inf)
                      + df.X1706.clip(0,np.inf)*(df.X1705/10000)*(df.X1703.isin([12,14,21,22,25,40,41,42,43,44,49,50,52,999])).astype(int)
                      + df.X1806.clip(0,np.inf)*(df.X1805/10000)*(df.X1803.isin([12,14,21,22,25,40,41,42,43,44,49,50,52,999])).astype(int)
                      + df.X2002.clip(0,np.inf) )
    else:
        df['oresre'] = (df[['X1405','X1409']].max(axis=1) + df[['X1505','X1509']].max(axis=1) + df.X1619.clip(0,np.inf)
                      + df[['X1605','X1609']].max(axis=1)
                      + df.X1706.clip(0,np.inf)*(df.X1705/10000)*(df.X1703.isin([12,14,21,22,25,40,41,42,43,44,49,50,52,999])).astype(int)
                      + df.X1806.clip(0,np.inf)*(df.X1805/10000)*(df.X1803.isin([12,14,21,22,25,40,41,42,43,44,49,50,52,999])).astype(int)
                      + df.X1906.clip(0,np.inf)*(df.X1905/10000)*(df.X1903.isin([12,14,21,22,25,40,41,42,43,44,49,50,52,999])).astype(int)
                      + df.X2002.clip(0,np.inf) )

    if year >= 2010:
        df['nnresre'] =(df.X1706.clip(0,np.inf)*(df.X1705/10000)*(df.X1703.isin([1,2,3,4,5,6,7,10,11,13,15,24,45,46,47,48,51,53,-7])).astype(int)
                      + df.X1806.clip(0,np.inf)*(df.X1805/10000)*(df.X1803.isin([1,2,3,4,5,6,7,10,11,13,15,24,45,46,47,48,51,53,-7])).astype(int)
                      + df.X2012.clip(0,np.inf)
                      - df.X1715*(df.X1705/10000)*(df.X1703.isin([1,2,3,4,5,6,7,10,11,13,15,24,45,46,47,48,51,53,-7])).astype(int)
                      - df.X1815*(df.X1805/10000)*(df.X1803.isin([1,2,3,4,5,6,7,10,11,13,15,24,45,46,47,48,51,53,-7])).astype(int)
                      - df.X2016 )
    else:
        df['nnresre'] =(df.X1706.clip(0,np.inf)*(df.X1705/10000)*(df.X1703.isin([1,2,3,4,5,6,7,10,11,13,15,24,45,46,47,48,51,53,-7])).astype(int)
                      + df.X1806.clip(0,np.inf)*(df.X1805/10000)*(df.X1803.isin([1,2,3,4,5,6,7,10,11,13,15,24,45,46,47,48,51,53,-7])).astype(int)
                      + df.X1906.clip(0,np.inf)*(df.X1905/10000)*(df.X1903.isin([1,2,3,4,5,6,7,10,11,13,15,24,45,46,47,48,51,53,-7])).astype(int)
                      + df.X2012.clip(0,np.inf)
                      - df.X1715*(df.X1705/10000)*(df.X1703.isin([1,2,3,4,5,6,7,10,11,13,15,24,45,46,47,48,51,53,-7])).astype(int)
                      - df.X1815*(df.X1805/10000)*(df.X1803.isin([1,2,3,4,5,6,7,10,11,13,15,24,45,46,47,48,51,53,-7])).astype(int)
                      - df.X1915*(df.X1905/10000)*(df.X1903.isin([1,2,3,4,5,6,7,10,11,13,15,24,45,46,47,48,51,53,-7])).astype(int)
                      - df.X2016 )

    df['flag781'] = 0
    df.loc[df.nnresre!=0, 'flag781'] = 1
    df.loc[df.nnresre!=0, 'nnresre'] = (df.nnresre - df.X2723*(df.X2710==78).astype(int) - df.X2740*(df.X2727==78).astype(int)
                                       - df.X2823*(df.X2810==78).astype(int) - df.X2840*(df.X2827==78).astype(int)
                                       - df.X2923*(df.X2910==78).astype(int) - df.X2940*(df.X2927==78).astype(int) )

    if year >= 2010:
        df['actbus'] = (df.X3129.clip(0,np.inf) + df.X3124.clip(0,np.inf) - df.X3126.clip(0,np.inf)*(df.X3127==5)
                      + df.X3121.clip(0,np.inf)*(df.X3122.isin([1,6])).astype(int) + df.X3229.clip(0,np.inf)
                      + df.X3224.clip(0,np.inf) - df.X3226.clip(0,np.inf)*(df.X3227==5).astype(int)
                      + df.X3221.clip(0,np.inf)*(df.X3222.isin([1,6])) + df.X3335.clip(0,np.inf)
                      + df.farmbus )

        df['nonactbus'] = (df.X3408.clip(0,np.inf) + df.X3412.clip(0,np.inf) + df.X3416.clip(0,np.inf)
                         + df.X3420.clip(0,np.inf) + df.X3452.clip(0,np.inf) + df.X3428.clip(0,np.inf) )

        df['bus'] = df.actbus + df.nonactbus
    else:

        df['actbus'] = (df.X3129.clip(0,np.inf) + df.X3124.clip(0,np.inf) - df.X3126.clip(0,np.inf)*(df.X3127==5)
                      + df.X3121.clip(0,np.inf)*(df.X3122.isin([1,6])).astype(int) + df.X3229.clip(0,np.inf)
                      + df.X3224.clip(0,np.inf) - df.X3226.clip(0,np.inf)*(df.X3227==5).astype(int)
                      + df.X3221.clip(0,np.inf)*(df.X3222.isin([1,6]).astype(int)) + df.X3329.clip(0, np.inf)
                      + df.X3324.clip(0,np.inf) - df.X3326.clip(0,np.inf)*(df.X3327==5).astype(int)
                      + df.X3321.clip(0,np.inf)*(df.X3322.isin([1,6])).astype(int) + df.X3335.clip(0,np.inf)
                      + df.farmbus )

        df['nonactbus'] = (df.X3408.clip(0,np.inf) + df.X3412.clip(0,np.inf) + df.X3416.clip(0,np.inf)
                         + df.X3420.clip(0,np.inf) + df.X3424.clip(0,np.inf) + df.X3428.clip(0,np.inf) )

        df['bus'] = df.actbus + df.nonactbus

    df['othnfin'] = df[['X4022', 'X4026', 'X4030', 'X4018']].sum(axis=1) - df.othfin

    df['nfin'] = df.vehic + df.houses + df.oresre + df.nnresre + df.bus + df.othnfin

    # Total nonfinancial assets excluding primary residence
    df['nhnfin'] = df.nfin - df.houses

    df['asset'] = df.fin + df.nfin

    df['heloc'] = 0
    df['mrthel'] = 0

    cond = (df.X1108+df.X1119+df.X1130)>=1

    df.loc[cond, 'heloc'] = ( df.X1108*(df.X1103==1).astype(int) + df.X1119*(df.X1114==1).astype(int)
                            + df.X1130*(df.X1125==1).astype(int) + df.X1136.clip(0,np.inf)*(df.X1108*(df.X1103==1).astype(int)
                            + df.X1119*(df.X1114==1).astype(int)
                            + df.X1130*(df.X1125==1).astype(int))/(df.X1108 + df.X1119 + df.X1130) )

    df.loc[cond, 'mrthel'] = ( df.X805 + df.X905 + df.X1005 + df.X1108*(df.X1103==1).astype(int)
                             + df.X1119*(df.X1114==1).astype(int) + df.X1130*(df.X1125==1).astype(int)
                             + df.X1136.clip(0,np.inf)*(df.X1108*(df.X1103==1).astype(int) + df.X1119*(df.X1114==1).astype(int)
                             + df.X1130*(df.X1125==1).astype(int))/(df.X1108+df.X1119+df.X1130) )

    df.loc[~cond, 'mrthel'] = df.X805 + df.X905 + df.X1005 + 0.5*(df.X1136.clip(0,np.inf)*(df.houses>0).astype(int))

    df['nhmort'] = df.mrthel - df.heloc

    df['othloc'] = 0

    df.loc[cond, 'othloc'] = (df.X1108*(df.X1103!=1).astype(int) + df.X1119*(df.X1114!=1).astype(int)
                            + df.X1130*(df.X1125!=1).astype(int) + df.X1136.clip(0,np.inf)*(df.X1108*(df.X1103!=1).astype(int))
                            + df.X1119*(df.X1114!=1).astype(int)
                            +(df.X1130*(df.X1125!=1).astype(int))/(df.X1108+df.X1119+df.X1130) )

    df.loc[~cond, 'othloc'] = ((df.houses<=0).astype(int) + 0.5*(df.houses>0).astype(int))*df.X1136.clip(0,np.inf)


    varlist = [12,14,21,22,25,40,41,42,43,44,49,50,52,53,999]

    df['mort1'] = df.X1715*df.X1705/10000*(df.X1703.isin(varlist).astype(int))
    df['mort2'] = df.X1815*df.X1805/10000*(df.X1803.isin(varlist).astype(int))
    df['mort3'] = 0

    if year <= 2007:
        df['mort3'] = df.X1915*df.X1905/10000*(df.X1903.isin(varlist).astype(int))
        df['resdbt'] = df.X1417 + df.X1517 + df.X1617 + df.X1621 + df.mort1 + df.mort2 + df.mort3 + df.X2006
    elif year == 2010:
        df['resdbt'] = df.X1417 + df.X1517 + df.X1621 + df.mort1 + df.mort2 + df.X2006
    elif year == 2013:
        df['resdbt'] = df.X1318 + df.X1337 + df.X1342 + df.mort1 + df.mort2 + df.X2006

    cond = (df.flag781!=1) & (df.oresre>0)
    df['flag782'] = 0
    df.loc[cond, 'flag782'] = 1
    df.loc[cond, 'resdbt'] = (df.resdbt + df.X2723*(df.X2710==78).astype(int) + df.X2740*(df.X2727==78).astype(int)
                           + df.X2823*(df.X2810==78).astype(int) + df.X2840*(df.X2827==78).astype(int)
                           + df.X2923*(df.X2910==78).astype(int) + df.X2940*(df.X2927==78).astype(int) )


    df['flag67'] = 0
    df.loc[df.oresre>0, 'flag67'] = 1
    df.loc[df.oresre>0, 'resdbt'] = (df.resdbt + df.X2723*(df.X2710==67).astype(int) + df.X2740*(df.X2727==67).astype(int)
                                   + df.X2823*(df.X2810==67).astype(int) + df.X2840*(df.X2827==67).astype(int)
                                   + df.X2923*(df.X2910==67).astype(int) + df.X2940*(df.X2927==67).astype(int) )

    if year >= 2010:
        df['ccbal'] = (df.X427.clip(0,np.inf) + df.X413.clip(0,np.inf) + df.X421.clip(0,np.inf)
                     + df.X430.clip(0,np.inf) + df.X7575.clip(0,np.inf) )

        df['noccbal'] = ((df.X427.clip(0,np.inf) + df.X413.clip(0,np.inf) + df.X421.clip(0,np.inf)
                       + df.X430.clip(0,np.inf) )==0)
    elif year >= 1992:
        df['ccbal'] = (df.X427.clip(0,np.inf) + df.X413.clip(0,np.inf) + df.X421.clip(0,np.inf)
                     + df.X430.clip(0,np.inf) + df.X424.clip(0,np.inf) + df.X7575.clip(0,np.inf) )

        df['noccbal'] = ((df.X427.clip(0,np.inf) + df.X413.clip(0,np.inf) + df.X421.clip(0,np.inf)
                       + df.X430.clip(0,np.inf) + df.X424.clip(0,np.inf) )==0)

    else:
        df['ccbal'] = (df.X427.clip(0,np.inf) + df.X413.clip(0,np.inf) + df.X421.clip(0,np.inf)
                     + df.X430.clip(0,np.inf) + df.X424.clip(0,np.inf) )

        df['noccbal'] = ((df.X427.clip(0,np.inf) + df.X413.clip(0,np.inf) + df.X421.clip(0,np.inf)
                       + df.X430.clip(0,np.inf) + df.X424.clip(0,np.inf) )==0)

    if year >= 1995:
        df['veh_inst'] = df.X2218 + df.X2318 + df.X2418 + df.X7169 + df.X2424 + df.X2519 + df.X2619 + df.X2625

        df['edn_inst'] =(df.X7824 + df.X7847 + df.X7870 + df.X7924 + df.X7947 + df.X7970 + df.X7179 +
                         df.X2723*(df.X2710==83).astype(int) + df.X2740*(df.X2727==83).astype(int)
                       + df.X2823*(df.X2810==83).astype(int) + df.X2840*(df.X2827==83).astype(int)
                       + df.X2923*(df.X2910==83).astype(int) + df.X2940*(df.X2927==83).astype(int) )

        df['install'] = (df.X2218 + df.X2318 + df.X2418 + df.X7169 + df.X2424 + df.X2519 + df.X2619 + df.X2625
                       + df.X7183 + df.X7824 + df.X7847 + df.X7870 + df.X7924 + df.X7947 + df.X7970 + df.X7179
                       + df.X1044 + df.X1215 + df.X1219 )
    elif year == 1992:
        df['veh_inst'] = df.X2218 + df.X2318 + df.X2418 +  df.X2424 + df.X2519 + df.X2619 + df.X2625

        df['edn_inst'] =(df.X7824 + df.X7847 + df.X7870 + df.X7924 + df.X7947 + df.X7970 +
                         df.X2723*(df.X2710==83).astype(int) + df.X2740*(df.X2727==83).astype(int)
                       + df.X2823*(df.X2810==83).astype(int) + df.X2840*(df.X2827==83).astype(int)
                       + df.X2923*(df.X2910==83).astype(int) + df.X2940*(df.X2927==83).astype(int) )

        df['install'] = (df.X2218 + df.X2318 + df.X2418 + df.X2424 + df.X2519 + df.X2619 + df.X2625
                       + df.X7824 + df.X7847 + df.X7870 + df.X7924 + df.X7947 + df.X7970 + df.X1044
                       + df.X1215 + df.X1219 )
    else:
        df['veh_inst'] = df.X2218 + df.X2318 + df.X2418 +  df.X2424 + df.X2519 + df.X2619 + df.X2625

        df['edn_inst'] =(df.X2723*(df.X2710==83).astype(int) + df.X2740*(df.X2727==83).astype(int)
                       + df.X2823*(df.X2810==83).astype(int) + df.X2840*(df.X2827==83).astype(int)
                       + df.X2923*(df.X2910==83).astype(int) + df.X2940*(df.X2927==83).astype(int) )

        df['install'] = (df.X2218 + df.X2318 + df.X2418 + df.X2424 + df.X2519 + df.X2619 + df.X2625
                       + df.X1044 + df.X1215 + df.X1219)

    cond = (df.flag781==0) & (df.flag782==0)
    df.loc[cond, 'install'] =(df.install +  df.X2723*(df.X2710==78).astype(int) + df.X2740*(df.X2727==78).astype(int)
                            + df.X2823*(df.X2810==78).astype(int) + df.X2840*(df.X2827==78).astype(int)
                            + df.X2923*(df.X2910==78).astype(int) + df.X2940*(df.X2927==78).astype(int) )

    df.loc[df.flag67==0, 'install'] =(df.install +  df.X2723*(df.X2710==67).astype(int) + df.X2740*(df.X2727==67).astype(int)
                            + df.X2823*(df.X2810==67).astype(int) + df.X2840*(df.X2827==67).astype(int)
                            + df.X2923*(df.X2910==67).astype(int) + df.X2940*(df.X2927==67).astype(int) )

    df.install +=(df.X2723*(~df.X2710.isin([67,78])).astype(int) + df.X2740*(~df.X2727.isin([67,78])).astype(int)
                + df.X2823*(~df.X2810.isin([67,78])).astype(int) + df.X2840*(~df.X2827.isin([67,78])).astype(int)
                + df.X2923*(~df.X2910.isin([67,78])).astype(int) + df.X2940*(~df.X2927.isin([67,78])).astype(int) )

    df['oth_inst'] = df.install - df.veh_inst - df.edn_inst

    if year == 1995:
        df['outmarg'] = df.X3932.clip(0,np.inf)*(df.X7194==5).astype(int)
    else:
        df['outmarg'] = df.X3932.clip(0,np.inf)

    if year >= 2010:
        df['outpen1']=df.X11027.clip(0,np.inf)*(df.X11070==5).astype(int)
        df['outpen2']=df.X11127.clip(0,np.inf)*(df.X11170==5).astype(int)
        df['outpen4']=df.X11327.clip(0,np.inf)*(df.X11370==5).astype(int)
        df['outpen5']=df.X11427.clip(0,np.inf)*(df.X11470==5).astype(int)
        df['outpen3']=0
        df['outpen6']=0
    elif year >= 2004:
        df['outpen1'] = df.X11027.clip(0,np.inf)*(df.X11070==5).astype(int)
        df['outpen2'] = df.X11127.clip(0,np.inf)*(df.X11170==5).astype(int)
        df['outpen3'] = df.X11227.clip(0,np.inf)*(df.X11270==5).astype(int)
        df['outpen4'] = df.X11327.clip(0,np.inf)*(df.X11370==5).astype(int)
        df['outpen5'] = df.X11427.clip(0,np.inf)*(df.X11470==5).astype(int)
        df['outpen6'] = df.X11527.clip(0,np.inf)*(df.X11570==5).astype(int)
    else:
        df['outpen1']=df.X4229.clip(0,np.inf)*(df.X4230==5).astype(int)
        df['outpen2']=df.X4329.clip(0,np.inf)*(df.X4330==5).astype(int)
        df['outpen3']=df.X4429.clip(0,np.inf)*(df.X4430==5).astype(int)
        df['outpen4']=df.X4829.clip(0,np.inf)*(df.X4830==5).astype(int)
        df['outpen5']=df.X4929.clip(0,np.inf)*(df.X4930==5).astype(int)
        df['outpen6']=df.X5029.clip(0,np.inf)*(df.X5030==5).astype(int)

    df['outpen'] = df[['outpen1', 'outpen2', 'outpen3', 'outpen4', 'outpen5', 'outpen6']].sum(axis=1)


    if year >= 2010:
        df['odebt'] = (df.outpen1 + df.outpen2 + df.outpen4 + df.outpen5 + df.X4010.clip(0,np.inf)
                     + df.X4032.clip(0,np.inf) + df.outmarg )
    else:
        df['odebt'] = (df.outpen1 + df.outpen2 + df.outpen3 + df.outpen4 + df.outpen5 + df.outpen6
                     + df.X4010.clip(0,np.inf) + df.X4032.clip(0,np.inf) + df.outmarg )

    df['debt'] = df.mrthel + df.resdbt + df.othloc + df.ccbal + df.install + df.odebt

    df['networth'] = df.asset - df.debt

    df['levratio'] = 0
    df.loc[(df.debt>0) & (df.asset>0), 'levratio'] = df.debt / df.asset
    df.loc[(df.debt>0) & (df.asset==0), 'levratio'] = 1

    df['debt2inc'] = 0
    df.loc[(df.debt>0) & (df.income>0), 'debt2inc'] = df.debt/df.income
    df.loc[(df.debt>0) & (df.income==0), 'debt2inc'] = 10

    return df
