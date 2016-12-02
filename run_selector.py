#!/usr/bin/python
import pickle
import time
import glob
from os.path import dirname, abspath
import os
import sys
from ROOT import TProof, TFileCollection, TChain, TDSet, TFileInfo

import ROOT

from selector import MyPySelector

import numpy as np

do_proof = True
do_proof = False

n_events = 5000


if do_proof :
    proof = TProof.Open('workers=15')
#elif n_events == -1 :
#    n_events = 10
    
#print 'path: ', sys.path


file_directory = '/home/llr/cms/pigard/ZZ_GEN_Analyzer/CMSSW_7_4_12/src/ZZjj_Analyzer/Ntuplizer/'


# select whether to do SR, CR or fake rate plots
mode = 'ZZ'
# luminosity in fb-1
lumi = 33.59
#lumi = 24.49
#lumi = 12.9



proc_postfix = '_full_v1'
#proc_postfix = '_30fb'


selections  = ['REG']#, 'RSE']#, 'TLE', 'RSE']

tree_names = { 'REG' : 'Tree/candTree', 'TLE' : 'Treetle/candTree', 'RSE' : 'TreelooseEle/candTree'}
mode_tree_name = {'ZZ' : 'ZZ', 'CRZL' : 'CRZL', 'CRZLL' : 'CRZLL', 'BKG' : 'CRZLL'}


xrootd_prefix = 'root://eosuser.cern.ch//eos//user/p/ppigard/cjlst_trees/160726/'
xrootd_prefix_own = 'root://eosuser.cern.ch//eos//user/p/ppigard/cjlst_trees/160916/'#cjlst_trees/'

data_prefix = ''

data_prefix = 'root://eosuser.cern.ch//eos/user/p/ppigard/cjlst_trees/160803_complete_newEbE/AllData/'
data_prefix = 'root://eosuser.cern.ch//eos/user/p/ppigard/cjlst_trees/160725/'
data_prefix = 'root://eosuser.cern.ch//eos/user/p/ppigard/cjlst_trees/160919_ICHEP_data/'
data_prefix = 'root://eosuser.cern.ch//eos/user/p/ppigard/cjlst_trees/full_dataset_v1/'
#if mode != 'ZZ' or 'RSE' in selections :  
#    data_prefix = 'root://eosuser.cern.ch//eos/user/p/ppigard/cjlst_trees/160725/AllData/'

#proc_postfix = '_24p49fb'datasets = [#    {'file_directory' : data_prefix, 'file_name' : 'full_dataset_v1/./AllData/ZZ4lAnalysis.root', 'tree_type' : 'CJLST_RECO', 'output_name' : 'AllData' + proc_postfix + '.root'},
datasets = [
#    {'file_directory' : data_prefix, 'file_name' : 'AllData/ZZ4lAnalysis.root', 'tree_type' : 'CJLST_RECO', 'output_name' : 'AllData' + proc_postfix + '.root'},
#    {'file_directory' : data_prefix, 'file_name' : 'AllDataPartH/ZZ4lAnalysis.root', 'tree_type' : 'CJLST_RECO', 'output_name' : 'AllData' + proc_postfix + '.root'},


    {'file_directory' : xrootd_prefix_own + 'ZZTo4l/', 'file_name' : 'ZZ4lAnalysis.root', 'tree_type' : 'CJLST_RECO', 'output_name' : 'ZZTo4l' + proc_postfix + '.root', 'isMC' : 1, 'k_factors' : ['KFactor_EW_qqZZ', 'KFactor_QCD_qqZZ_M']},
    {'file_directory' : xrootd_prefix_own + 'ZZTo4lamcatnlo/', 'file_name' : 'ZZ4lAnalysis.root', 'tree_type' : 'CJLST_RECO', 'output_name' : 'ZZTo4l' + proc_postfix + '.root', 'isMC' : 1, 'k_factors' : ['KFactor_EW_qqZZ', 'KFactor_QCD_qqZZ_M']},



#    {'file_directory' : xrootd_prefix_own + 'ZZjj_ewk/', 'file_name' : 'ZZ4lAnalysis.root', 'tree_type' : 'CJLST_RECO', 'output_name' : 'ZZjj_ewk' + proc_postfix + '.root', 'isMC' : 1, 'k_factors' : [], 'options' : ['create_out_tree']},
    {'file_directory' : xrootd_prefix_own + 'ZZjj_qcd/', 'file_name' : 'ZZ4lAnalysis.root', 'tree_type' : 'CJLST_RECO', 'output_name' : 'ZZjj_qcd' + proc_postfix + '.root', 'isMC' : 1, 'k_factors' : [], 'options' : ['create_out_tree']},
#
#
#  # VBF PHANTOM
    {'file_directory' : xrootd_prefix_own + 'VBFTo4eJJ_0PMH125Contin_phantom128/', 'file_name' : 'ZZ4lAnalysis_partial.root', 'tree_type' : 'CJLST_RECO', 'output_name' : 'VBFTo4eJJ_0PMH125Contin_phantom128' + proc_postfix + '.root', 'isMC' : 1, 'k_factors' : []},
    {'file_directory' : xrootd_prefix_own + 'VBFTo2e2muJJ_0PMH125Contin_phantom128/', 'file_name' : 'ZZ4lAnalysis.root', 'tree_type' : 'CJLST_RECO', 'output_name' : 'VBFTo2e2muJJ_0PMH125Contin_phantom128' + proc_postfix + '.root', 'isMC' : 1, 'k_factors' : []},
    {'file_directory' : xrootd_prefix_own + 'VBFTo4muJJ_0PMH125Contin_phantom128/', 'file_name' : 'ZZ4lAnalysis.root', 'tree_type' : 'CJLST_RECO', 'output_name' : 'VBFTo4muJJ_0PMH125Contin_phantom128' + proc_postfix + '.root', 'isMC' : 1, 'k_factors' : []},
#
#
#
#
#   # gg > ZZ > 4l MCFM
    {'file_directory' : xrootd_prefix_own + 'ggTo4e_Contin_MCFM701/', 'file_name' : 'ZZ4lAnalysis.root', 'tree_type' : 'CJLST_RECO', 'output_name' : 'ggTo4e_Contin_MCFM701' + proc_postfix + '.root', 'isMC' : 1, 'k_factors' : ['KFactor_QCD_ggZZ_Nominal'], 'options' : ['create_out_tree']},
    {'file_directory' : xrootd_prefix_own + 'ggTo4mu_Contin_MCFM701/', 'file_name' : 'ZZ4lAnalysis.root', 'tree_type' : 'CJLST_RECO', 'output_name' : 'ggTo4mu_Contin_MCFM701' + proc_postfix + '.root', 'isMC' : 1, 'k_factors' : ['KFactor_QCD_ggZZ_Nominal'], 'options' : ['create_out_tree']},
    {'file_directory' : xrootd_prefix_own + 'ggTo2e2mu_Contin_MCFM701/', 'file_name' : 'ZZ4lAnalysis.root', 'tree_type' : 'CJLST_RECO', 'output_name' : 'ggTo2e2mu_Contin_MCFM701' + proc_postfix + '.root', 'isMC' : 1, 'k_factors' : ['KFactor_QCD_ggZZ_Nominal'], 'options' : ['create_out_tree']},
#    {'file_directory' : xrootd_prefix_own + 'ggTo2e2tau_Contin_MCFM701/', 'file_name' : 'ZZ4lAnalysis.root', 'tree_type' : 'CJLST_RECO', 'output_name' : 'ggTo2e2tau_Contin_MCFM701' + proc_postfix + '.root', 'isMC' : 1, 'k_factors' : ['KFactor_QCD_ggZZ_Nominal']},
#    {'file_directory' : xrootd_prefix_own + 'ggTo2mu2tau_Contin_MCFM701/', 'file_name' : 'ZZ4lAnalysis.root', 'tree_type' : 'CJLST_RECO', 'output_name' : 'ggTo2mu2tau_Contin_MCFM701' + proc_postfix + '.root', 'isMC' : 1, 'k_factors' : ['KFactor_QCD_ggZZ_Nominal']},
#    {'file_directory' : xrootd_prefix_own + 'ggTo4tau_Contin_MCFM701/', 'file_name' : 'ZZ4lAnalysis.root', 'tree_type' : 'CJLST_RECO', 'output_name' : 'ggTo4tau_Contin_MCFM701' + proc_postfix + '.root', 'isMC' : 1, 'k_factors' : ['KFactor_QCD_ggZZ_Nominal']},
###
###
####    # Irreducible background
    {'file_directory' : xrootd_prefix_own + 'TTZ_MLM/', 'file_name' : 'ZZ4lAnalysis.root', 'tree_type' : 'CJLST_RECO', 'output_name' : 'TTZ_MLM' + proc_postfix + '.root', 'isMC' : 1, 'k_factors' : []},
###
###    {'file_directory' : xrootd_prefix + 'TTZ/', 'file_name' : 'ZZ4lAnalysis.root', 'tree_type' : 'CJLST_RECO', 'output_name' : 'TTZ' + proc_postfix + '.root', 'isMC' : 1, 'k_factors' : []},
    {'file_directory' : xrootd_prefix_own + 'WWZ/', 'file_name' : 'ZZ4lAnalysis.root', 'tree_type' : 'CJLST_RECO', 'output_name' : 'WWZ' + proc_postfix + '.root', 'isMC' : 1, 'k_factors' : []},
#
    {'file_directory' : xrootd_prefix + 'WZTo3LNu/', 'file_name' : 'ZZ4lAnalysis.root', 'tree_type' : 'CJLST_RECO', 'output_name' : 'WZTo3LNu' + proc_postfix + '.root', 'isMC' : 1},
##
##
##
##
#    {'file_directory' : xrootd_prefix + 'DYJetsToLL_M50/', 'file_name' : 'ZZ4lAnalysis.root', 'tree_type' : 'CJLST_RECO', 'output_name' : 'DYJetsToLL_M50' + proc_postfix + '.root', 'isMC' : 0.0004404},
###    {'file_directory' : '', 'file_name' : 'ttbar_12p9_ZZ4lAnalysis.root', 'tree_type' : 'CJLST_RECO', 'output_name' : 'TTJets' + proc_postfix + '.root', 'isMC' : 0.0004404},

 



#    {'file_directory' : '/data/DATA/temp_pigard/TLE/8X_Zgamma' + '/', 'file_name' : '8X_Zgamma_ZZ4lAnalysis.root', 'tree_type' : 'CJLST_RECO', 'output_name' : 'Zg.root', 'isMC' : 1},
           ]



for i in range(len(datasets)) :
    if do_proof :
        proof.ClearInput()

    theSet = datasets[i]
    isMC = False
    if 'isMC' in theSet : isMC = True

    file_directory = theSet['file_directory']
    file_list = []
    if 'file_name' in theSet :
        file_list.append(file_directory + theSet['file_name'])
    else :
        file_list = glob.glob(file_directory)


    print 'Files to be run over: ', file_list
    if isMC :

        fi = ROOT.TFile.Open(file_list[0])
        histo = fi.Get("ZZTree/Counters")
        sumOfWeights = histo.GetBinContent(40)
        print 'sumOfWeights ', sumOfWeights
    for sel in selections :
        print 'Runnning over selection ', sel
        #print 'Loading tree ', mode + tree_names[sel]
        
        ch = TChain(mode_tree_name[mode] + tree_names[sel]) #'ntuplizer/tree')
        params = {'outputName' : sel + '_' + mode + '_' + theSet['output_name'], 'tree_type' : mode_tree_name[mode] + tree_names[sel], 'selection' : sel, 'mode' : mode}# theSet['tree_type']}
        if 'options' not in theSet: theSet['options'] = []
        params['options'] = theSet['options']

        params['sklearn_est'] = 'SKL_minimal_retrain.pkl'#'SKL_3var.pkl'#'SKL_minimal_retrain.pkl'#'SKL_minimal.pkl'
#        params['sklearn_est'] = 'SKL_noRelPt.pkl' 
        params['isMC'] = isMC
        if isMC :
            params['sumOfWeights'] = sumOfWeights
            params['lumi'] = lumi
            params['pu_weight'] = 'data/pu_weights_v1.root'
            if 'k_factors' in theSet :
                params['k_factors'] = theSet['k_factors']
            else :
                params['k_factors'] = []        
        
        libs = sys.path
        params['python_libraries'] = libs


        if mode == 'BKG' :
            params['fake_ratio_files'] = {'fake_ratio_file' : 'fake_ratios' + proc_postfix} 

        params['cwd'] = os.getcwd()
        #print params['python_libraries']
        with open('/tmp/curr_selector_params.pkl', 'wb') as pkl:
#        with open('/home/llr/cms/pigard/Moriond17_plots/Histogrammer/test.pkl', 'wb') as pkl:
            pickle.dump(params, pkl)


        for f in file_list :
            ch.Add(f)
    
        time.sleep(1)  # needed for GUI to settle
    
   
    
        if do_proof :
            ch.SetProof()
            #proof.Load('/home/llr/cms/pigard/background_estimates/parameters.pkl')
    
            proof.Exec('TPython::Exec("%s");' %
                ("import sys; sys.path.insert(0,'"+dirname(abspath("selector.py"))+"')"))
    
    
    
        #print 'About to start selector'
        print ch.Process( 'TPySelector', 'selector')# , n_events)
    
        #proof.Process(dataset, 'TPySelector', 'selector')
#    raw_input("Press Enter to continue...")
#proof.Process(fc, 'TPySelector', 'selector')
