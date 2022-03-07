import numpy as np
import joblib


def add_laplacian_noise(sensitivity, effective_eps):
    myscale = sensitivity/effective_eps
    noise = np.random.laplace(0., myscale, 1)
    return noise

def add_gamma_noise(sensitivity=1, effective_eps=0.5,n=113):
    myscale = sensitivity/effective_eps
    myshape = 1/n
    rnd = np.random.gamma(myshape, myscale, 2)
    return rnd[0] - rnd[1]

def get_noisy_granular_counts(granular_counts):
    noisy_granular_counts = []
    for val in granular_counts:
        noisy_granular_counts.append((val + add_laplacian_noise(sensitivity, effect_eps))[0])
    return noisy_granular_counts

def compute_all_stats(s_granular_counts):
    cs0 = s_granular_counts[2] + s_granular_counts[3]
    cs1 = s_granular_counts[0] + s_granular_counts[1]
    cy0 = s_granular_counts[0] + s_granular_counts[2]
    cy1 = s_granular_counts[1] + s_granular_counts[3]
    tot = cy0 + cy1
    return cs0,cs1,cy0,cy1,tot

def fair_balance_class(all_stats):
    weights = []
    weights.append(all_stats['cs1y0'])
    weights.append(all_stats['cs1y1'])
    weights.append(all_stats['cs0y0'])
    weights.append(all_stats['cs0y1'])

    for i,w in enumerate(weights):
        weights[i] = 1/weights[i]

    return weights

def class_balance(all_stats):
    weights = []
    weights.append(all_stats['tot']/(2*all_stats['cy0']))
    weights.append(all_stats['tot']/(2*all_stats['cy1']))
    return weights

def reweighing(all_stats):
    weights = []
    w = (all_stats['cs1'] * all_stats['cy0']) / (all_stats['tot'] * all_stats['cs1y0'])
    weights.append(w)
    w = (all_stats['cs1'] * all_stats['cy1']) / (all_stats['tot'] * all_stats['cs1y1'])
    weights.append(w)
    w = (all_stats['cs0'] * all_stats['cy0']) / (all_stats['tot'] * all_stats['cs0y0'])
    weights.append(w)
    w = (all_stats['cs0'] * all_stats['cy1']) / (all_stats['tot'] * all_stats['cs0y1'])
    weights.append(w)
    return weights



epsilon = 1
COMPUTE_WT_FUNC = {'fbclass': fair_balance_class, 'skcb': class_balance, 'rw': reweighing}

# Adding noise to denominators : Method 2
sensitivity = 1
effect_eps = epsilon/2 # Each user has contribution for 2 variables
n = 113 # number of parties

# We want to add noise to 4 cs1y0, cs1y1, cs0y0, cs0y1
# count(y and a)
#TODO: Change this to dictionary
granular_counts = [14795, 8457, 2245, 1623] #cs1y0, cs0y0, cs1y1, cs0y1
#print(sum(granular_counts))

print('{:30s}{}'.format('Granular counts', granular_counts))

noisy_granular_counts = get_noisy_granular_counts(granular_counts)
print('{:30s}{}'.format('Noisy granular counts', noisy_granular_counts))

cs0_,cs1_,cy0_,cy1_,tot_ = compute_all_stats(granular_counts)
print('{:30s}{},{},{},{}'.format('Stats', cs0_,cs1_,cy0_,cy1_,tot_))

cs0,cs1,cy0,cy1,tot = compute_all_stats(noisy_granular_counts)
print('{:30s}{},{},{},{}'.format('Noisy Stats', cs0,cs1,cy0,cy1,tot))

all_stats = {'cs1y0':granular_counts[0], 'cs1y1':granular_counts[2],
             'cs0y0':granular_counts[1], 'cs0y1':granular_counts[3],
             'cs0':cs0_, 'cs1':cs1_,
             'cy0':cy0_, 'cy1':cy1_,
             'tot':tot_}

all_stats_noisy = {'cs1y0':noisy_granular_counts[0], 'cs1y1':noisy_granular_counts[2],
             'cs0y0':noisy_granular_counts[1], 'cs0y1':noisy_granular_counts[3],
             'cs0':cs0, 'cs1':cs1,
             'cy0':cy0, 'cy1':cy1,
             'tot':tot}


print("======================================================================")
# Distributed Laplacian Noise
SZ = [239, 235, 236, 206, 127, 222, 138, 206, 211, 228, 197, 226, 238,
      207, 203, 237, 229, 169, 237, 227, 232, 235, 218, 231, 203, 169,
      231, 205, 224, 238, 231, 232, 220,  74, 224, 217, 211, 238, 239,
      233, 238, 226, 166, 175, 217, 216, 131, 209, 157, 152, 233, 185,
      237, 179, 237, 233, 233, 239, 168, 233, 234, 189, 131, 184, 207,
      216, 211, 184, 206, 211, 235]
print('{:30s}'.format('Distributed Laplacian Noise - For cs0y0 only'))
#cs0y0_dict = joblib.load("cs0y0_dict.joblib") #{key= userid : value=cs0y0} count of only negative lables only for female
cs0y0_agg_true = 0
cs0y0_agg_dln = 0
for counts in SZ:
    cs0y0_agg_true = cs0y0_agg_true + counts
    counts_noisy = counts + add_gamma_noise(1,0.5,113)
    cs0y0_agg_dln = cs0y0_agg_dln + counts_noisy

print('{:30s}{}'.format('True aggregated',cs0y0_agg_true))
print('{:30s}{}'.format('DistributedLN aggregated',cs0y0_agg_dln))



print("======================================================================")
# Execute FB Class Technique
print(">>>>>>> Computing FBClass")
weights = COMPUTE_WT_FUNC.get('fbclass', lambda: 'Invalid')(all_stats)
noisy_weights = COMPUTE_WT_FUNC.get('fbclass', lambda: 'Invalid')(all_stats_noisy)

print('{:30s}{}'.format('FBCLASS WTS', weights))
print('{:30s}{}'.format('FBCLASS NOISY WTS', noisy_weights))

print("Normalizing FBClass")
sum_wts = 4 # rounded off
normalized_const = (all_stats['tot'] / sum_wts)
print('{:30s}{}'.format('Norm. FBCLASS WTS',np.array(weights)*normalized_const))
normalized_const = (all_stats_noisy['tot'] / sum_wts)
print('{:30s}{}'.format('Norm.FBCLASS NOISY WTS', np.array(noisy_weights)*normalized_const))

print("======================================================================")
# Execute Sklearn CB Technique
print(">>>>>>> Computing CB")
weights = COMPUTE_WT_FUNC.get('skcb', lambda: 'Invalid')(all_stats)
noisy_weights = COMPUTE_WT_FUNC.get('skcb', lambda: 'Invalid')(all_stats_noisy)

print('{:30s}{}'.format('CB WTS', weights))
print('{:30s}{}'.format('CB NOISY WTS', noisy_weights))

print("======================================================================")
# Execute RW Technique
print(">>>>>>> Computing Reweighing technique")
weights = COMPUTE_WT_FUNC.get('rw', lambda: 'Invalid')(all_stats)
noisy_weights = COMPUTE_WT_FUNC.get('rw', lambda: 'Invalid')(all_stats_noisy)

print('{:30s}{}'.format('RW WTS', weights))
print('{:30s}{}'.format('RW NOISY WTS', noisy_weights))





'''
inv_denominators,noisy_inv_denominators =[], []
for val,noisy_val in zip(denominators,noisy_denominators):
    inv_denominators.append(1/val)
    noisy_inv_denominators.append(1/noisy_val)


print('{:30s}{}'.format('Denominators', denominators))
print('{:30s}{}'.format('Noisy Denom', noisy_denominators))
print('{:30s}{}'.format('1/Denom', inv_denominators))
print('{:30s}{}'.format('1/Noisy Denom', noisy_inv_denominators))

# Adding Laplacian noise to final float values
weights = [6.75904022e-05, 1.18245241e-04, 4.45434298e-04, 6.16142945e-04]
weights_normalized = [0.45826293, 0.80170273, 3.02004454, 4.17744917]

len_y = 27120
sum_wts = 3.9999999999993388
normalized_const = (len_y / sum_wts)
noisy_inv_denominators = np.array(noisy_inv_denominators)
noisy_inv_denominators = noisy_inv_denominators * normalized_const

print('{:30s}{}'.format('Norm. wts', weights_normalized))
print('{:30s}{}'.format('Norm. Noisy wts', weights_normalized))
#print("1/Denom wts",weights)
#print("Norm. wts",weights_normalized)
'''
