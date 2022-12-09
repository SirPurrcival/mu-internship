# Packages
import scipy.io as sp

def data_extraction(monkey, session, contrastcomparison, trial):
    data = sp.loadmat('M'+str(monkey)+'_session_'+str(session)+'.mat')

    data = data['alldata']

    data = data[0, 0]

    actual_data = data[0] # ML data 1x9 cell
    # rfdata = data[1] # ML 1x1 struct same level
    # probe1_contacts = data[2] # ML other fields of the same 1x6 struct 'alldata'
    # probe2_contacts = data[3] # ___"____
    # probe3_contacts = data[4] # ___"____
    # eyedata = data[5] # ___"____

    cont_data = actual_data[0, contrastcomparison] # ML one of the 1x1 structs from the 9 options
    CSD = cont_data[0, 0][0] # ML CSD 1x1 struct
    probe1_contrast = cont_data[0, 0][1] # ML other levels of same cont_data structure, defining the contrasts of the individual 3 probes for that particular comparison
    probe2_contrast = cont_data[0, 0][2] # ___"___
    probe3_contrast = cont_data[0, 0][3] # ___"___
    probe_contrasts = {
        'Probe_1': probe1_contrast[0,0],
        'Probe_2': probe2_contrast[0,0],
        'Probe_3': probe3_contrast[0,0]
        }

    trial_timeseries = CSD[0,0][3][0, trial-1]
    # time = CSD[0,0][2][0, trial-1]
    # print(np.shape(time))
    # print(np.shape(X))

    # plt.figure()
    # plt.subplot(311)
    # plt.imshow(X[0:15], aspect='auto')
    # plt.subplot(312)
    # plt.imshow(X[16:31], aspect='auto')
    # plt.subplot(313)
    # plt.imshow(X[32:47], aspect='auto')
    # plt.xticks(np.linspace(0, np.shape(time)[1], 8),
    #             np.round(np.linspace(0, np.shape(time)[1], 8),2))
    # plt.show()

    return trial_timeseries, probe_contrasts

def shift(time_series):
    time_series1 = time_series[:,0:-1]
    time_series2 = time_series[:,1:]

    return time_series1, time_series2
