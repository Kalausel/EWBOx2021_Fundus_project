# -*- coding: utf-8 -*-
"""
This function converts the keywords given to indicate the eye diseases to one of
8 categories given in 'annotations.csv'.
"""

def keywords2vec_helper(keywords):
    '''
    Input:
        string
        This is the string unaltered from the label file.
    Returns
        list
        List of length 8 with entry 1 at position of positive diagnoses
    '''

    # These are the 8 different categories as extracted in Test_keywords.py
    categories=[['normal fundus'],
     ['moderate non proliferative retinopathy',
      'mild nonproliferative retinopathy',
      'diabetic retinopathy',
      'severe nonproliferative retinopathy',
      'proliferative diabetic retinopathy',
      'severe proliferative diabetic retinopathy'],
     ['glaucoma'],
     ['cataract'],
     ['wet age-related macular degeneration',
      'dry age-related macular degeneration',
      'age-related macular degeneration'],
     ['hypertensive retinopathy'],
     ['pathological myopia',
      'myopia retinopathy',
      'myopic maculopathy',
      'myopic retinopathy'],
     ['laser spot',
      'branch retinal artery occlusion',
      'macular epiretinal membrane',
      'epiretinal membrane',
      'drusen',
      'vitreous degeneration',
      'retinal pigmentation',
      'myelinated nerve fibers',
      'rhegmatogenous retinal detachment',
      'depigmentation of the retinal pigment epithelium',
      'abnormal pigment',
      'post laser photocoagulation',
      'spotted membranous change',
      'macular hole',
      'epiretinal membrane over the macula',
      'central retinal artery occlusion',
      'pigment epithelium proliferation',
      'atrophy',
      'chorioretinal atrophy',
      'white vessel',
      'retinochoroidal coloboma',
      'atrophic change',
      'retinitis pigmentosa',
      'retina fold',
      'branch retinal vein occlusion',
      'optic disc edema',
      'retinal pigment epithelium atrophy',
      'refractive media opacity',
      'microvascular anomalies',
      'central retinal vein occlusion',
      'tessellated fundus',
      'maculopathy',
      'oval yellow-white atrophy',
      'retinal vascular sheathing',
      'macular coloboma',
      'vessel tortuosity',
      'idiopathic choroidal neovascularization',
      'wedge-shaped change',
      'optic nerve atrophy',
      'wedge white line change',
      'old chorioretinopathy',
      'punctate inner choroidopathy',
      'old choroiditis',
      'chorioretinal atrophy with pigmentation proliferation',
      'congenital choroidal coloboma',
      'optic disk epiretinal membrane',
      'morning glory syndrome',
      'retinal pigment epithelial hypertrophy',
      'old branch retinal vein occlusion',
      'asteroid hyalosis',
      'retinal artery macroaneurysm',
      'suspicious diabetic retinopathy',
      'glial remnants anterior to the optic disc',
      'vascular loops',
      'diffuse chorioretinal atrophy',
      'optic discitis',
      'intraretinal hemorrhage',
      'pigmentation disorder',
      'arteriosclerosis',
      'silicone oil eye',
      'choroidal nevus',
      'old central retinal vein occlusion',
      'diffuse retinal atrophy',
      'fundus laser photocoagulation spots',
      'abnormal color of  optic disc',
      'vitreous opacity',
      'macular pigmentation disorder',
      'macular epimacular membrane',
      'peripapillary atrophy',
      'retinal detachment',
      'central serous chorioretinopathy',
      'post retinal laser surgery',
      'intraretinal microvascular abnormality']]

    # First, retrieve the keywords as  single strings in a list.
    single_keywords=keywords.split(',')
    single_keywords=[s.replace('suspected','').replace('suspicious','').strip() for s in single_keywords]
    for k in ['lens dust', 'low image quality', 'image offset', 'no fundus image',
              'anterior segment image', 'optic disk photographically invisible']:
        if k in single_keywords:
            single_keywords.remove(k) # remove irrevelant keywords
    vector=[0,0,0,0,0,0,0,0] # "empty" vector
    # For each keyword, set the corresponding entry to 1
    for keyword in single_keywords:
        single_vector=[keyword in category for category in categories]
        vector=list(map(lambda x,y: x or y, single_vector, vector))
    # If none of the disease keywords has been found, set to 'normal fundus'
    # This is the case, if the only keyword is one of the list above with
    # 'lens dust', 'low image quality', and so on
    if vector==[0,0,0,0,0,0,0,0]:
        vector=[1,0,0,0,0,0,0,0]
    return [int(x) for x in vector] # Convert True to 1, False to 0



# This function does the same for either a single eye or for both eyes.
def keywords2vec(keywords, both_sides=False):
    '''
    Parameters
    ----------
    keywords :     string or list of strings (length 2)
                   This is the string unaltered from the label file.
                   If both_sides==True, pass list of two strings
    both_sides :   bool, optional
                   The default is False.
                   If True, return vector for patient (both eyes)
    Returns
    -------
        list
        List of length 8 with entry 1 at position of positive diagnoses
    '''

    if both_sides is False:
        if isinstance(keywords, str) is False:
            raise TypeError('Expected input, if optional arg both_sides==False:'+
                            'string \nExpected input, if optional arg both_sides'+
                            '==True: list of strings; length 2')
        return keywords2vec_helper(keywords)

    if both_sides is True:
        if isinstance(keywords, list) is False or len(keywords) != 2:
            raise TypeError('Expected input, if optional arg both_sides==False:'+
                            ' string \nExpected input, if optional arg both_sides'+
                            '==True: list of strings; length 2')
        vec=[0,0,0,0,0,0,0,0]
        normal=[0,0] # indicate that one eye is healthy (normal fundus)
        for k in range(2): #  loop through both eyes
            newvec=keywords2vec_helper(keywords[k]) # call my own fct to convert to vector
            if newvec==[1,0,0,0,0,0,0,0] or newvec==[0,0,0,0,0,0,0,0]:
                normal[k]=1
            vec=list(map(lambda x,y: x or y, vec, newvec)) # collect diagnoses from both eyes
        # If exactly one eye is healthy, have to set "normal fundus" to zero.
        if sum(normal)==1:
            vec[0]=0
        # If both eyes are healthy, set vec manually, because some lines do not contain the word
        # 'normal fundus', even though the eyes are healthy.
        elif sum(normal)==2:
            vec=[1,0,0,0,0,0,0,0]
        return vec
