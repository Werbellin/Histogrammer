### selector module (selector.py, name has to match as per in main.py)
from ROOT import TPySelector, gROOT, TH1F, TH1I, TProfile2D, TH2F, TLorentzVector, TH1, TEfficiency, TH2, TH3, TProfile

import ROOT


from array import array
import pickle

from math import *

from sklearn.externals import joblib
from sklearn.ensemble import GradientBoostingClassifier

import numpy as np


libs = []#args['python_libraries']


#print libs 

#libs = ['/grid_mnt/vol__vol_U__u/llr/cms/pigard/ZZjj_generator_study', '/opt/exp_soft/llr/root/v6.06.00-el6-gcc48/lib/root', '/opt/exp_soft/llr/python/2.7.10/lib/python2.7', '/opt/exp_soft/llr/python/2.7.10/lib/python2.7/site-packages', '/opt/rh/devtoolset-2/root/usr/lib64/python2.6/site-packages', '/opt/rh/devtoolset-2/root/usr/lib/python2.6/site-packages', '/usr/lib64/python', '/opt/exp_soft/llr/python/2.7.10/lib/python27.zip', '/opt/exp_soft/llr/python/2.7.10/lib/python2.7/plat-linux2', '/opt/exp_soft/llr/python/2.7.10/lib/python2.7/lib-tk', '/opt/exp_soft/llr/python/2.7.10/lib/python2.7/lib-old', '/opt/exp_soft/llr/python/2.7.10/lib/python2.7/lib-dynload', '/home/llr/cms/pigard/.local/lib/python2.7/site-packages']
#libs = ['/Users/pigard/CMS/ZZ_analysis', '/Users/pigard/Envs/default_venv/lib/python2.7/site-packages/root_numpy-4.4.0.dev0-py2.7-macosx-10.11-x86_64.egg', '/Users/pigard/ROOT/install_dir_v6-05-02/lib', '/Users/pigard/Envs/default_venv/lib/python27.zip', '/Users/pigard/Envs/default_venv/lib/python2.7', '/Users/pigard/Envs/default_venv/lib/python2.7/plat-darwin', '/Users/pigard/Envs/default_venv/lib/python2.7/plat-mac', '/Users/pigard/Envs/default_venv/lib/python2.7/plat-mac/lib-scriptpackages', '/Users/pigard/Envs/default_venv/lib/python2.7/lib-tk', '/Users/pigard/Envs/default_venv/lib/python2.7/lib-old', '/Users/pigard/Envs/default_venv/lib/python2.7/lib-dynload', '/usr/local/Cellar/python/2.7.10_2/Frameworks/Python.framework/Versions/2.7/lib/python2.7', '/usr/local/Cellar/python/2.7.10_2/Frameworks/Python.framework/Versions/2.7/lib/python2.7/plat-darwin', '/usr/local/Cellar/python/2.7.10_2/Frameworks/Python.framework/Versions/2.7/lib/python2.7/lib-tk', '/usr/local/Cellar/python/2.7.10_2/Frameworks/Python.framework/Versions/2.7/lib/python2.7/plat-mac', '/usr/local/Cellar/python/2.7.10_2/Frameworks/Python.framework/Versions/2.7/lib/python2.7/plat-mac/lib-scriptpackages', '/Users/pigard/Envs/default_venv/lib/python2.7/site-packages']
import sys
import array

from math import acos, cos

libs_to_add = [item for item in libs if item not in sys.path]
for lib in libs_to_add :
    sys.path.append(lib)


def get_p4(event, index, do_jet = False) :
    p4 = ROOT.TLorentzVector()
    if do_jet :
        p4.SetPtEtaPhiM(event.JetPt[index], event.JetEta[index], event.JetPhi[index], event.JetMass[index])
    else :
        p4.SetPtEtaPhiM(event.LepPt[index], event.LepEta[index], event.LepPhi[index], 0.)

    return p4

def get_N_jets(tree) :
    leptons = [(tree.LepEta[i], tree.LepPhi[i]) for i in range(len(tree.LepEta))]
    jets = [(tree.JetEta[i], tree.JetPhi[i]) for i in range(len(tree.JetEta)) if abs(tree.JetEta[i]) < 4.7 and tree.JetPt[i] > 30.]

    n_jets = 0
    for j in jets :
        has_overlap = False
        for l in leptons :
            if ((l[0] - j[0])**2 + (acos(cos(l[1] - j[1])))**2 )**0.5 < 0.3 :
                has_overlap = True

        if not has_overlap : n_jets += 1

    return n_jets


def get_k_factors(tree, k_factor_names) :
    k_factor = 1.

    for name in k_factor_names :
        if name == 'KFactor_QCD_qqZZ_M' :
            k_factor *= tree.KFactor_QCD_qqZZ_M
            continue
        if name == 'KFactor_EW_qqZZ' :
            k_factor *= tree.KFactor_EW_qqZZ
            continue
        if name == 'KFactor_QCD_ggZZ_Nominal' :
            k_factor *= tree.KFactor_QCD_ggZZ_Nominal
            continue
        sys.exit("ERROR: K factor not found!")         
        
    return k_factor

def get_failing_index(tree, debug=False) :
    failing_indeces = []
    for i, isID in enumerate(tree.LepisID) :
        if debug : print 'PDGID ', tree.LepLepId[i], ' iso ', tree.LepCombRelIsoPF[i], '  isID ', bool(isID), ' SIP ', tree.LepSIP[i] 
        if(tree.LepCombRelIsoPF[i] > 0.35 or not bool(isID)) : 
            failing_indeces.append(i)
            continue
    #print 'failing ', failing_indeces
    return failing_indeces  

def test_bit(mask, bit) :
    return (mask >> bit) & 1 
 



def get_TLE_index(tree) :
    for i, ID in enumerate(tree.LepLepId) :
#     print 'id', ID
        if abs(ID) == 22 :
            return i

def get_TLE_dR_Z(tree) :
    TLE_index = get_TLE_index(tree)

    lep_1_i = 0
    lep_2_i = 1

    if TLE_index <= 1 :
        lep_1_i = 2
        lep_2_i = 3

    lep_1 = ROOT.TLorentzVector()
    lep_1.SetPtEtaPhiM(tree.LepPt[lep_1_i], tree.LepEta[lep_1_i], tree.LepPhi[lep_1_i], 0)
    lep_2 = ROOT.TLorentzVector()
    lep_2.SetPtEtaPhiM(tree.LepPt[lep_2_i], tree.LepEta[lep_2_i], tree.LepPhi[lep_2_i], 0)

    lep_TLE = ROOT.TLorentzVector()
    lep_TLE.SetPtEtaPhiM(tree.LepPt[TLE_index], tree.LepEta[TLE_index], tree.LepPhi[TLE_index], 0)

    return lep_TLE.DeltaR(lep_1 + lep_2)
        

def get_FR_type(tree, index) :
    abs_id = abs(tree.LepLepId[index])
    if abs_id == 13 : return 'MUON'
    if abs_id == 22 : return 'TLE'
    if abs_id == 11 and tree.LepSIP[index] < 4. : return 'REG'
    if abs_id == 11 and tree.LepSIP[index] >= 4. : return 'RSE'
    sys.exit("ERROR: Lepton type not recosgnized") 
 

def try_except(fn):
    """decorator for extra debugging output"""
    def wrapped(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except:
            import traceback
            traceback.print_exc()
            assert(0)
    return wrapped


class MyPySelector(ROOT.TPySelector):

#    def getZ(self, tree) :
#        z = []
#        for i in range(len(tree.genParticle_px)) :
#            if abs(tree.genParticle_pdgId[i]) in [23] :
#                p4 = TLorentzVector(tree.genParticle_px[i], tree.genParticle_py[i], tree.genParticle_pz[i], tree.genParticle_e[i])
#                z.append(Particle((tree.genParticle_pdgId[i], p4)))
#                #z.append(Particle((tree.genParticle_pdgId[i], tree.genParticle_px[i], tree.genParticle_py[i], tree.genParticle_pz[i], tree.genParticle_e[i], tree.genParticle_mass[i])))
#        return z

#    def getPartons(self, tree) :
#        jets = []
#        for i in range(len(tree.genJetAK4_pt)) :
#             if abs(tree.genParticle_pdgId[i]) in [1, 2, 3, 4, 5, 6, 9, 21] :
#                p4 = TLorentzVector(tree.genParticle_px[i], tree.genParticle_py[i], tree.genParticle_pz[i], tree.genParticle_e[i])
#
#                jets.append(Particle((tree.genParticle_pdgId[i], p4)))
#
#               #jets.append(Particle((tree.genParticle_pdgId[i], tree.genParticle_px[i], tree.genParticle_py[i], tree.genParticle_pz[i], tree.genParticle_e[i], tree.genParticle_mass[i])))
#            
#        return jets



    plots = {}
    def __init__(self):
        #print self.__class__.__module__+": init"
        with open('/tmp/curr_selector_params.pkl', 'read') as pkl:
#        with open('/home/llr/cms/pigard/Moriond17_plots/Histogrammer/test.pkl', 'read') as pkl:
            args = pickle.load(pkl)
        #print 'Settings: ', args
        self.cwd = args['cwd']
        self.outputName = args['outputName']
        self.selection = args['selection']
        self.mode = args['mode']
        self.isMC = args['isMC']
        if self.isMC and self.mode == 'BKG' :
            print 'Switching to CRZLL mode for MC'
            self.mode = 'CRZLL'   
 
        if self.isMC :
            self.sumOfWeights = args['sumOfWeights']
            self.lumi = args['lumi']
            self.k_factor_names = args['k_factors']
            #print self.k_factor_names
            self.pu_weight_name = None
            if 'pu_weight' in args :
                self.pu_weight_name = args['pu_weight']

        self.REG_FR_TEff = None
        self.RSE_FR_TEff = None
        self.TLE_FR_TEff = None
        self.fake_ratios = {}
        self.pt_fr_max = 80.
        self.sklearn_est = '/home/llr/cms/pigard/bdt_trainings/' + args['sklearn_est']
        self.est = None
        if self.sklearn_est is not '' :
            self.est = joblib.load(self.sklearn_est)
            if not isinstance(self.est, GradientBoostingClassifier) :
                print 'Failed to load the sklearn estimator'
                self.est = None

        self.create_out_tree = False
        if 'create_out_tree' in args['options'] :
            self.create_out_tree = True

        self.skip_odd_events = False
        #if 'skip_odd_events' in args['options'] :
        #    self.skip_odd_events = True


        if self.mode == 'BKG' :
            self.fake_ratio_files = args['fake_ratio_files']

    def Begin(self):
        #print 'Ran Begin()'
        return
    @try_except
    def SlaveBegin(self, tree):
        #print 'SlaveBegin'
        #print 'nevents: ', self.nevents
        
        #print args.libs

        etaBins = ['EB', 'EE',]# 'eta_0_0.4', 'eta_0.4_0.8', 'eta_0.8_1.48']
        #self.out_tree = Tree("MVA_tree", model=Event)
        if self.create_out_tree :
            self.out_tree = ROOT.TTree("out_tree","ExampleTree")

            self.EventNumber = array.array( 'i', [ 0 ] )
            self.final_weight = array.array( 'd', [ 0 ] )

            self.m_4l = array.array( 'd', [ 0 ] )
            self.m_jj = array.array( 'd', [ 0 ] )
            self.dEta_tj = array.array( 'd', [ 0 ] )
            self.Z1_zepp = array.array( 'd', [ 0 ] )
            self.Z2_zepp = array.array( 'd', [ 0 ] )
            self.rel_pt_hard = array.array( 'd', [ 0 ] )
            self.tj_delta_rel = array.array( 'd', [ 0 ] )
            self.tj1_eta_x_tj2_eta = array.array( 'd', [ 0 ] )
 
            self.out_tree.Branch("EventNumber",self.EventNumber,"EventNumber/I")
            self.out_tree.Branch("final_weight",self.final_weight, "final_weight/D")

            self.out_tree.Branch("m_4l",self.m_4l, "m_4l/D")
            self.out_tree.Branch("m_jj",self.m_jj,"m_jj/D")
            self.out_tree.Branch("dEta_tj",self.dEta_tj,"dEta_tj/D")
            self.out_tree.Branch("Z1_zepp",self.Z1_zepp,"Z1_zepp/D")
            self.out_tree.Branch("Z2_zepp",self.Z2_zepp,"Z2_zepp/D")
            self.out_tree.Branch("rel_pt_hard",self.rel_pt_hard,"rel_pt_hard/D")
            self.out_tree.Branch("tj_delta_rel",self.tj_delta_rel,"tj_delta_rel/D")
            self.out_tree.Branch("tj1_eta_x_tj2_eta",self.tj1_eta_x_tj2_eta,"tj1_eta_x_tj2_eta/D")

            self.GetOutputList().Add(self.out_tree) 

        TH1.SetDefaultSumw2()
        TH2.SetDefaultSumw2()
        TH3.SetDefaultSumw2()
        self.plots = {}

        self.plots['n_events'] = TH1F('n_events', ';n_events;Entries', 10, 0., 10.)
        self.plots['lep_n'] =  TH1F('lep_n', ';# leptons;Entries', 7,0, 7.0)

        self.plots['weights'] = TH1F('weights', ';n_events;Entries', 10, 0, 10)
        self.plots['skim_n_events'] = TH1F('skim_n_events', ';n_events;Entries', 10, 0, 10)

        self.plots['skim_weights'] = TH1F('skim_weights', ';weights;Entries', 10, 0, 10)

        #self.plots['weights'].SetCanExtend(TH1.kAllAxes)

        kinematic_selections = ['', 'HZZ']
        regions = []
        regions = ['ZZ', 'HZZ', 'BLS', 'VBS', 'nVBS']

        #if self.mode == 'ZZ' :
        #        regions += ['SR', 'HZZ', 'BLS', 'VBS', 'nVBS']

        if self.mode in ['CRZLL', 'BKG'] :
            #    for region in ['2P2F', '3P1F', '3P1F_from_2F'] :
            #        for kin_sel in ['', 'HZZ'] :
            kin_selections = [k + cr if r == '' else r + '_' + cr for cr in ['2P2Fuw','3P1Fuw', '2P2Fw', '3P1Fw', '3P1F_from_2F'] for r in regions]
            print 'kin', kin_selections 
            regions +=  kin_selections #[kin_sel + region for kin_sel in ['', 'HZZ_', 'BLS_'] for region in ['2P2F', '3P1F', '3P1F_from_2F']]


#        kinematic_selections = ['ZZ']
#        print regions
        if self.mode in ['ZZ', 'CRZLL', 'BKG'] :
#        if self.mode == 'ZZ' or self.mode == 'BKG' :

            #if self.mode in ['CRZLL', 'BKG'] :
            #    bins_ZZMass = 50
            #else : 
            bins_ZZMass = 100
            for region in regions :
                for fs in ['4e','4m', '2e2m', '2m2e','all','mix', 'unidentified'] :
                    #self.plots['ZZMass_' + fs] = TH1F('ZZMass_' + fs, ';m_{4l} [GeV];Entries', 100, 0, 1000)
                    self.plots[region + '_Z1Mass_' + fs] = TH1F(region + '_Z1Mass_' + fs, ';m_{Z1} [GeV];Entries', 60, 60, 120)
                    self.plots[region + '_Z2Mass_' + fs] = TH1F(region + '_Z2Mass_' + fs, ';m_{Z1} [GeV];Entries', 60, 60, 120)

                    if region.find('HZZ') == 0 :
                        self.plots[region + '_ZZMass_' + fs] = TH1F(region + '_ZZMass_' + fs, ';m_{4l} [GeV];Entries', 204, 70, 886)
                    else :
                        self.plots[region + '_ZZMass_' + fs] = TH1F(region + '_ZZMass_' + fs, ';m_{4l} [GeV];Entries', bins_ZZMass, 0, 1000)

                self.plots[region + '_Nvtx'] = TH1F(region + '_Nvtx', ';N_{vtx};Entries', 50, 0, 50)

                self.plots[region + '_Njets_4p7'] = TH1F(region + '_Njets_4p7', ';N jets (|#eta_{j}| < 4.7);Entries', 5, 0, 5)
                self.plots[region + '_Njets_4p7_cleaned'] = TH1F(region + '_Njets_4p7_cleaned', ';N jets (|#eta_{j}| < 4.7);Entries', 5, 0, 5)
                self.plots[region + '_Njets_4p7_int'] = TH1F(region + '_Njets_4p7_int', ';N jets (|#eta_{j}| < 4.7);Entries', 5, 0, 5)
                for i in range (1, 6) : self.plots[region + '_Njets_4p7_int'].GetXaxis().SetBinLabel(i, str(i - 1) if i < 5 else '#geq 4')
                self.plots[region + '_Njets_2p4'] = TH1F(region + '_Njets_2p4', ';N jets (|#eta_{j}| < 2.4);Entries', 5, 0, 5)
                self.plots[region + '_Njets_2p4_int'] = TH1F(region + '_Njets_2p4_int', ';N jets (|#eta_{j}| < 2.4);Entries', 5, 0, 5)
                for i in range (1, 6) : self.plots[region + '_Njets_2p4_int'].GetXaxis().SetBinLabel(i, str(i - 1) if i < 5 else '#geq 4')
                self.plots[region + '_DiJetMass_4p7'] = TH1F(region + '_DiJetMass_4p7', ';m_{jj} [GeV];Entries', 16, 0, 1600)
                #self.plots[region + '_DiJetMass_PFMET'] = TH2F(region + '_DiJetMass_PFMET', ';m_{jj} [GeV]; PF MET', 15, 0, 1500, 0, 30, 60, 99999)
                self.plots[region + '_DiJetDEta_4p7'] = TH1F(region + '_DiJetDEta_4p7', ';|#Delta#eta_{jj}|;Entries', 14, 0, 7)
                self.plots[region + '_jet_1_pt'] = TH1F(region + '_jet_1_pt', ';p_{T}^{jet1} [GeV];Entries', 6, array.array('d', [30, 50, 100, 200, 300, 400, 500]))
                self.plots[region + '_jet_2_pt'] = TH1F(region + '_jet_2_pt', ';p_{T}^{jet2} [GeV];Entries', 6, array.array('d', [30, 50, 100, 200, 300, 400, 500]))
                self.plots[region + '_jet_1_abseta'] = TH1F(region + '_jet_1_abseta', ';#eta^{jet1};Entries', 3, array.array('d', [0, 1.5, 3, 4.7]))
                self.plots[region + '_jet_2_abseta'] = TH1F(region + '_jet_2_abseta', ';#eta^{jet2};Entries', 3, array.array('d', [0, 1.5, 3, 4.7]))
                self.plots[region + '_jet_1_eta'] = TH1F(region + '_jet_1_eta', ';#eta^{jet1};Entries', 12, -6, 6)
                self.plots[region + '_jet_2_eta'] = TH1F(region + '_jet_2_eta', ';#eta^{jet2};Entries', 12, -6, 6)
     
                self.plots[region + '_Z_1_zepp'] = TH1F(region + '_Z_1_zepp', ';z^{*}_{Z1};Entries', 12, -6, 6)
                self.plots[region + '_Z_2_zepp'] = TH1F(region + '_Z_2_zepp', ';z^{*}_{Z2};Entries', 12, -6, 6)
                self.plots[region + '_rel_pt_hard'] = TH1F(region + '_rel_pt_hard', ';p_{T}^{rel., hard};Entries', 10, 0, 1)
                self.plots[region + '_delta_rel'] = TH1F(region + '_delta_rel', ';p_{T}^{rel., jets};Entries', 25, 0, 1)
                self.plots[region + '_ZZjj_MVA'] = TH1F(region + '_ZZjj_MVA', ';BDT score;Entries', 20, -1, 1)
                self.plots[region + '_sklearn_MVA'] = TH1F(region + '_sklearn_MVA', ';BDT score;Entries', 20, -1, 1)
#                self.plots[region + '_sklearn_MVA'] = TH1F(region + '_sklearn_MVA', ';BDT score;Entries', 20, -1, 1)

                self.plots[region + '_sklearn_MVA_fine'] = TH1F(region + '_sklearn_MVA_fine', ';BDT score;Entries', 40, -1, 1)
                self.plots[region + '_sklearn_MVA_SR'] = TH1F(region + '_sklearn_MVA_SR', ';BDT score;Entries', 10, 0.5, 1)
                self.plots[region + '_sklearn_MVA_proba'] = TH1F(region + '_sklearn_MVA_proba', ';BDT score;Entries', 20, 0, 1)

                self.plots[region + '_ZZjj_MVA_blind'] = TH1F(region + '_ZZjj_MVA_blind', ';BDT score;Entries', 20, -1, 1)

                self.plots[region + '_ZZjj_MVA_fine'] = TH1F(region + '_ZZjj_MVA_fine', ';BDT score;Entries', 100, -1, 1)

                self.plots[region + '_costhetastar'] = TH1F(region + '_costhetastar', ';cos #theta*;Entries', 20, -1, 1)
                self.plots[region + '_helphi'] = TH1F(region + '_helphi', ';#varphi*;Entries', 20, -3.3, 3.3)
                self.plots[region + '_helcosthetaZ1'] = TH1F(region + '_helcosthetaZ1', ';cos #theta_{Z1};Entries', 20, -1, 1)
                self.plots[region + '_helcosthetaZ2'] = TH1F(region + '_helcosthetaZ2', ';cos #theta_{Z2};Entries', 20, -1, 1)
                self.plots[region + '_phistarZ1'] = TH1F(region + '_phistarZ1', ';#varphi*;Entries', 20, -3.3, 3.3)

#                self.plots[region + '_'] = TH1F(region + '_', ';cos #theta*;Entries', 100, -1, 1)0v$y

       
        #if self.mode == 'CRZLL' or self.mode == 'BKG' :
        #    for fs in ['Zp2e', 'Zp2m', '4e','4m', '2e2m', '2m2e', 'all', 'unidentified'] :
        #        self.plots['ZZMass_' + fs] = TH1F('ZZMass_' + fs, ';weights;Entries', 200, 0, 2000)
        #        regions = ['2P2F', '3P1F', 'SS', 'none']
        #        if self.selection == 'RSE' : 
        #            #print 'Adding regions'
        #            regions += ['2P2F_ss_lt4', '2P2F_ss_gt4', '2P2F_os_gt4', '2P2F_os_lt4']
        #            regions += ['3P1F_ss_lt4', '3P1F_ss_gt4', '3P1F_os_gt4', '3P1F_os_lt4']

        #        for sel in regions :
        #            #print 'Histo ', sel + '_ZZMass_' + fs
        #            self.plots[sel + '_ZZMass_' + fs] = TH1F(sel + '_ZZMass_' + fs, ';M_{4l};Entries', 50, 0, 1000)
        #            self.plots[sel + '_fZMass_' + fs] = TH1F(sel + '_fZMass_' + fs, ';M_{Z};Entries', 75, 0, 150)
        #            self.plots[sel + '_Z1Mass_' + fs] = TH1F(sel + '_Z1Mass_' + fs, ';M_{Z};Entries', 75, 0, 150)

        #if self.mode == 'BKG' :
        #    #print 'Histos BKG'
        #    for fs in ['4e','4m', '2e2m', '2m2e'] :
        #        #self.plots['BKG_ZZMass_' + fs] = TH1F('BKG_ZZMass_' + fs, ';weights;Entries', 200, 0, 2000)
        #        regions = ['2P2F', '3P1F', '3P1F_from_2F', 'SS']
        #        if self.selection == 'RSE' : 
        #            #print 'Adding regions'
        #            regions += ['2P2F_ss_lt4', '2P2F_ss_gt4', '2P2F_os_gt4']
        #            regions += ['3P1F_ss_lt4', '3P1F_ss_gt4', '3P1F_os_gt4']
        #            regions += ['3P1F_ss_lt4_from_2F', '3P1F_ss_gt4_from_2F', '3P1F_os_gt4_from_2F']

        #        for sel in regions :
        #            #print 'Histo ', 'BKG_'  + sel + '_ZZMass_' + fs
        #            self.plots['BKG_'  + sel + '_ZZMass_' + fs] = TH1F('BKG_' + sel + '_ZZMass_' + fs, ';M_{4l};Entries', 50, 0, 1000)
        #            self.plots['BKG_'  + sel + '_ZZMass_FR_' + fs] = TH2F('BKG_' + sel + '_ZZMass_FR_' + fs, ';M_{4l};Entries', 50, 0, 1000, 1000, 0, .1)
        #            self.plots['BKG_'  + sel + '_ZZMass_FR_profile_' + fs] = TProfile('BKG_' + sel + '_ZZMass_FR_profile_' + fs, ';M_{4l};Entries', 50, 0, 1000, 0, 0.1)

                    #self.plots[sel + '_fZMass_' + fs] = TH1F(sel + '_fZMass_' + fs, ';M_{Z};Entries', 75, 0, 150)



        if self.mode == 'CRZL' :
            a = array.array( 'd', [5, 7, 10, 20, 30, 45, self.pt_fr_max] )
            ptl3_bins = array.array( 'd', [7, 10, 20, 30, 45, self.pt_fr_max] )
            eta_muon = array.array( 'd', [0, 1.2, 2.4])
            eta_ele = array.array( 'd', [0, 1.479, 2.5])

            for fs in ['2epe', '2epm','2mpm', '2mpe', 'Zpe', 'Zpm'] :
                for eta_bin in ['EB', 'EE'] :

                    self.plots[fs + '_Z1_mass_all_' + eta_bin] = TH1F(fs + '_Z1_mass_all_' + eta_bin, ';M_{Z1} [GeV];Entries', 20, 60., 120.)
                    self.plots[fs + '_Z1_mass_pass_' + eta_bin] = TH1F(fs + '_Z1_mass_pass_' + eta_bin, ';M_{Z1} [GeV];Entries', 20, 60., 120.)
                    self.plots[fs + '_event_ETmiss_all_' + eta_bin] = TH1F(fs + '_event_ETmiss_all_' + eta_bin, ';E_{T_{miss}} [GeV];Entries', 10, 0., 50.)
                    self.plots[fs + '_event_ETmiss_pass_' + eta_bin] = TH1F(fs + '_event_ETmiss_pass_' + eta_bin, ';E_{T_{miss}} [GeV];Entries', 10, 0., 50.)

                    self.plots[fs + '_ptl3_all_' + eta_bin] = TH1F(fs + '_ptl3_all_' + eta_bin, ';third lepton p_{T} [GeV];Entries', len(ptl3_bins)-1, ptl3_bins)
                    self.plots[fs + '_ptl3_pass_' + eta_bin] = TH1F(fs + '_ptl3_pass_' + eta_bin, ';third lepton p_{T} [GeV];Entries', len(ptl3_bins)-1, ptl3_bins)


                    self.plots[fs + '_TLE_dR_Z_' + eta_bin] = TH1F(fs + '_TLE_dR_Z_' + eta_bin, ';M_{Z};Entries', 60, 0, 6)

                    self.plots[fs + '_fakeRatio_' + eta_bin] = TEfficiency(fs + '_fakeRatio_' + eta_bin, '; p_{T} [GeV]; Fake rate (Reco #rightarrow ID+ISO)', len(a) - 1, a)  
                    self.plots[fs + '_NoMET_fakeRatio_' + eta_bin] = TEfficiency(fs + '_NoMET_fakeRatio_' + eta_bin, '; p_{T} [GeV]; Fake rate (Reco #rightarrow ID+ISO)', len(a) - 1, a)  
                    self.plots[fs + '_fakeRatio_' + eta_bin + '_mZ1'] = TEfficiency(fs + '_fakeRatio_' + eta_bin + '_mZ1', '; m_{Z1} [GeV]; Fake rate (Reco #rightarrow ID+ISO)',30 ,60, 120)  
                    self.plots[fs + '_fakeRatio_from_REG_' + eta_bin] = TEfficiency(fs + '_fakeRatio_from_REG_' + eta_bin, '; p_{T} [GeV]; Fake rate (Reco #rightarrow ID+ISO)', len(a) - 1, a)  
                    self.plots[fs + '_fakeRatio_from_REG_DoubleM2l_' + eta_bin] = TEfficiency(fs + '_fakeRatio_from_REG_DoubleM2l_' + eta_bin, '; p_{T} [GeV]; Fake rate (Reco #rightarrow ID+ISO)', len(a) - 1, a)  

                if fs[-1] == 'e' : eta = eta_ele
                if fs[-1] == 'm' : eta = eta_muon
                self.plots[fs + '_fakeRatio'] = TEfficiency(fs + '_fakeRatio', '; p_{T} [GeV];#eta; Fake rate (Reco #rightarrow ID+ISO)', len(a) - 1, a, len(eta) - 1 , eta)
                self.plots[fs + '_fakeRatio_from_REG'] = TEfficiency(fs + '_fakeRatio_from_REG', '; p_{T} [GeV];#eta; Fake rate (Reco #rightarrow ID+ISO)', len(a) - 1, a, len(eta) - 1 , eta)
 
                self.plots[fs + '_ptl3_all'] = TH2F(fs + '_ptl3_all', ';third lepton p_{T} [GeV];Entries', len(ptl3_bins)-1, ptl3_bins, len(eta) - 1 , eta)
                self.plots[fs + '_ptl3_pass'] = TH2F(fs + '_ptl3_pass', ';third lepton p_{T} [GeV];Entries', len(ptl3_bins)-1, ptl3_bins, len(eta) - 1 , eta)

        for name, plot in self.plots.items() :

            #try :
            #     = getattr(l, 'Scale')
            #print plot
            #plot.Sumw2()
            self.GetOutputList().Add(plot)

        if self.mode == 'BKG' :
            use_corrected = True
            version_name = 'AllData_12p9fb'
            use_corrected = {'MUON' : True, 'REG' : True, 'RSE' : False, 'TLE' : False}
            for sel in ['MUON' ,'REG', 'RSE']: #, 'TLE'] :
                teff_name = {'MUON' : 'Zpm', 'REG' : 'Zpe', 'RSE' : 'Zpe', 'TLE' : 'Zpe'}
                file_name = {'MUON' : 'REG', 'REG' : 'REG', 'RSE' : 'RSE', 'TLE' : 'TLE'}
                corrected_names = {'MUON' : 'corrected_muon_fake_ratio', 'REG' : 'corrected_electron_fake_ratio'}
                #if sel == 'TLE' : continue

                if sel == 'RSE' :
                    if use_corrected[sel] : 
                        f = ROOT.TFile.Open('/home/llr/cms/pigard/background_estimates/fake_ratios_v2.root')
                        f.Get('corrected_electron_fake_ratio')
                        print f
                        
                    else :
                        f = ROOT.TFile.Open('/home/llr/cms/pigard/background_estimates/REG_CRZL_' + version_name + '.root')
                        f.cd('raw')
                        #f.ls()
                        eff = f.Get('raw/Zpe_fakeRatio_from_REG')
                else : 
                    if use_corrected[sel] : 
                        f = ROOT.TFile.Open('%s/%s.root'%(self.cwd, self.fake_ratio_files['fake_ratio_file']))
                        #f = ROOT.TFile.Open('/home/llr/cms/pigard/background_estimates/' + 'fake_ratios_v2.root')
                        #print 'Selection ', sel
                        eff = f.Get(corrected_names[sel])
                        eff.SetDirectory(0)
                        #print eff
                        import types
                        def GetEfficiency(self, _bin):
                            #print 'self=',self
                            #print 'bin=', _bin
                            return self.GetBinContent(_bin)
                        eff.GetEfficiency = types.MethodType(GetEfficiency, eff)
                        test_bin = eff.FindFixBin(60, 2.3)
                        #print 'bin ', test_bin
                        #print 'Test fake ratio ', eff.GetEfficiency(eff.FindFixBin(60, 2.3)) 
                    else :
                        f = ROOT.TFile.Open('/home/llr/cms/pigard/background_estimates/' + file_name[sel] + '_CRZL_' + version_name + '.root')
                        f.cd('raw')
                        #f.ls()
                        #eff = f.Get('raw/Zpe_fakeRatio_from_REG')
                        eff = f.Get('raw/' + teff_name[sel]  + '_fakeRatio')
                #print eff 


                self.fake_ratios[sel] = eff

            print self.fake_ratios

        self.pu_weight_histo = None
        if self.isMC and self.pu_weight_name is not None :
            f = ROOT.TFile.Open('%s/%s'%(self.cwd, self.pu_weight_name), 'read')
            self.pu_weight_histo = f.Get('weights')
            self.pu_weight_histo.SetDirectory(0)
            self.pu_weight_histo.Scale(1./self.pu_weight_histo.Integral())    


    @try_except
    def Process(self, entry):
        #print 'in Process!!!'
        #print 'chain?', self.fChain.GetEntries()
        self.fChain.GetEntry(entry)
         
        tree = self.fChain
        
        total_weight = 1.0
        curr_k_factor = 1.0
        if self.isMC :
            curr_k_factor = get_k_factors(tree, self.k_factor_names)
            total_weight = tree.xsec * curr_k_factor * (tree.overallEventWeight / float(self.sumOfWeights)) * self.lumi * 1000.
            if self.pu_weight_histo is not None :
                pu_weight = self.pu_weight_histo.GetBinContent(self.pu_weight_histo.FindBin(tree.NTrueInt))         
                print 'reweighing by ', pu_weight
                total_weight *= pu_weight
        self.plots['n_events'].Fill(1)

        final_state = 'unidentified'
        regions = {}

        if True : #self.mode == 'ZZ' or self.mode == 'CRZLL' or self.mode == 'BKG' :
            final_state_id = abs(tree.Z1Flav*tree.Z2Flav)
            if(final_state_id == 121*242 or final_state_id == 121*121) : final_state = "4e"
            if(final_state_id == 169*242 or final_state_id == 169*121) : final_state = "2e2m"
            if(final_state_id == 169*169) : final_state = "4m"
            if final_state == '2e2m' and abs(tree.Z2Flav) != 169 : final_state = '2m2e'

            #self.plots['ZZMass_' + final_state].Fill(tree.ZZMass, total_weight) 

            pass_SR = False
            low_Z_mass = 60
            up_Z_mass = 120
            pass_ZZsel = (low_Z_mass <= tree.Z1Mass <= up_Z_mass) and (low_Z_mass <= tree.Z2Mass <= up_Z_mass) # tree.ZZsel >=120
            pass_BLSsel = pass_ZZsel and tree.DiJetMass > 100.
            pass_VBSsel = pass_BLSsel and tree.DiJetMass > 400. and abs(tree.DiJetDEta) > 2.4

            pass_HZZsel = tree.ZZsel >= 90
            pass_REG = True
            pass_TLE = False
            pass_RSE = True
            if self.selection == 'TLE' :
                TLE_index = get_TLE_index(tree)
                if TLE_index is None :
                    sys.exit("ERROR: Did not find TLE")
                    return 0
                if tree.ZZsel==120 and TLE_dR_Z > 1.6 and tree.LepPt[TLE_index] > 30. : pass_TLE = True 
 

            pass_SR = (self.selection == 'REG' and pass_REG) or (self.selection == 'RSE' and pass_RSE) or (self.selection == 'TLE' and pass_TLE)
            print pass_ZZsel 
            pass_selection = False

            #if self.mode == 'ZZ' :

            pass_selection = pass_SR and pass_ZZsel
            if pass_selection : 
                regions['ZZ'] = total_weight
            if pass_SR and pass_HZZsel :
                regions['HZZ'] = total_weight
            if pass_SR and pass_BLSsel :
                regions['BLS'] = total_weight / curr_k_factor
            if pass_SR and pass_VBSsel :
                regions['VBS'] = total_weight / curr_k_factor
            if pass_SR and pass_BLSsel and not pass_VBSsel:
                regions['nVBS'] = total_weight / curr_k_factor 

            print 'regions1 ', regions


            control_regions = {}
            if self.mode in  ['CRZLL', 'BKG'] :
                CRZLLss = 21
                CRZLLos_2P2F = 22
                CRZLLos_3P1F = 23
                
                bit_flag = 'none'

                failing_indeces = []
               
                if(test_bit(tree.CRflag, CRZLLos_2P2F) and test_bit(tree.CRflag, CRZLLos_3P1F)) : 
                    print 'Event number: ', tree.EventNumber , ' run number ', tree.RunNumber
                    get_failing_index(tree, True)
                    sys.exit("ERROR: event is both 3P1F and 2P2F!")
                if test_bit(tree.CRflag, CRZLLss) : bit_flag = "SS"
                if test_bit(tree.CRflag, CRZLLos_2P2F) : bit_flag = "2P2F"
                if test_bit(tree.CRflag, CRZLLos_3P1F) : bit_flag = "3P1F"
                #print selection
                if bit_flag == 'SS' : return 1


                pass_3P1F = False
                pass_2P2F = False
                failZ = 'none'
                failLep = get_failing_index(tree)
                if len(failLep) < 2 and bit_flag == "2P2F" :
                    print 'ERROR: CR Bit says 2P2F but only one failing lepton found'
                    print 'failing indeces ', failLep
                    #print 'resetting to 2,3'
                    #failLep = [2, 3]
                    if self.selection != 'RSE' :
                        sys.exit("ERROR: CR Bit says 2P2F but only one failing lepton found")
                    else :
                        if len(failLep) == 1 :
                            bit_flag = "3P1F" 
                        else : return 0
                    #get_failing_index(tree, debug=True)
                if len(failLep) > 1 and bit_flag == "3P1F" :
                    sys.exit("ERROR: CR Bit says 3P1F but more than one failing lepton found")

                if len(failLep) < 1 and bit_flag == "3P1F" :
                    print 'ERROR: CR Bit says 3P1F but only one failing lepton found'
                    print 'failing indeces ', failLep
                    if self.selection != 'RSE' :
                        sys.exit("ERROR: CR Bit says 2P2F but only one failing lepton found")
                    else :
                        return 0
                    #get_failing_index(tree, debug=True)
                if len(failLep) > 1 and bit_flag == "3P1F" :
                    sys.exit("ERROR: CR Bit says 3P1F but more than one failing lepton found")


                if self.selection == 'RSE' :
                    failZ = 'Z2'
                else :
                    failZ = 'Z2' if failZ[0] > 1 else 'Z1'

                abs_fZ_id = abs(getattr(tree, failZ + 'Flav'))
                CR_final_state = ''
                if abs_fZ_id == 121 or abs_fZ_id == 242: CR_final_state = 'Zp2e'
                if abs_fZ_id == 169 : CR_final_state = 'Zp2m'


 
                failZMass = getattr(tree, failZ + 'Mass')
                passfZMass = False
                if 60 < tree.Z1Mass < 120 and 60 < tree.Z2Mass < 120 : passfZMass = True 

                if self.selection == 'REG' :
                    pass_2P2F = bit_flag == '2P2F'
                    pass_3P1F = bit_flag == '3P1F'

                if self.selection == 'RSE' :
                    if passfZMass : pass_2P2F = bit_flag == '2P2F'
                    if passfZMass : pass_3P1F = bit_flag == '3P1F'

                if self.selection == 'TLE' :
                    TLE_index = get_TLE_index(tree)
                    if TLE_index is None :
#                        print "ERROR: Did not find TLE"
                        sys.exit("ERROR: Did not find TLE")
                    TLE_dR_Z = get_TLE_dR_Z(tree)
                    if TLE_dR_Z > 1.6 and tree.LepPt[TLE_index] > 30. : 
                        if passfZMass : pass_2P2F = bit_flag == '2P2F'
                        if passfZMass : pass_3P1F = bit_flag == '3P1F'
                pass_ss = tree.LepLepId[2] * tree.LepLepId[3] > 0
                pass_SIP_gt_4 = tree.LepSIP[2] >= 4.0 or tree.LepSIP[3] >= 4.0

                # in RSE we dont want the CR taht is part of the regular analsyis
                skip_CR = False
                if self.selection == 'RSE' :
                    if (not pass_ss) and (not pass_SIP_gt_4) : skip_CR = True 
                
                if pass_2P2F and not skip_CR :
                    control_regions['2P2Fuw'] = 1.0# .append('2P2Fuw')
#                    regions['HZZ_2P2F'] = total_weight
#                    if passfZMass :
#                        regions['2P2F'] = total_weight

                if pass_3P1F and not skip_CR :
                    control_regions['3P1Fuw'] = 1.0
                    #regions['HZZ_3P1F'] = total_weight
#                    if passfZMass :
#                        regions['3P1F'] = total_weight

                
                   

            if self.mode == 'BKG' :
                #print 'RUNNING BKG MODE'
                if pass_2P2F or pass_3P1F :
                    weight_1 = 1.
                    weight_2 = 1.
                    fr_1 = 1.
                    fr_2 = 1.
                    if pass_3P1F or pass_2P2F :
                        #print 'pass_3P1F ', pass_3P1F, '  pass_2P2F ', pass_2P2F 
                        #print failLep
                        lep_1_index = failLep[0]
                        lep_1_pt = tree.LepPt[lep_1_index]
                        lep_1_abs_eta = abs(tree.LepEta[lep_1_index])
                        if lep_1_pt > self.pt_fr_max : lep_1_pt = self.pt_fr_max - 0.5
                        lep_1_teff = self.fake_ratios[get_FR_type(tree, lep_1_index)]
                        #print 'Looking up ', get_FR_type(tree, lep_1_index), '  eta ', lep_1_abs_eta, ' pt ', lep_1_pt
                        fr_1 = lep_1_teff.GetEfficiency(lep_1_teff.FindFixBin(lep_1_pt, lep_1_abs_eta))
                        #print 'Fake ratio: ', fr_1
                    if pass_2P2F :
                        #print 'failLep ', failLep
                        lep_2_index = failLep[1]
                        lep_2_pt = tree.LepPt[lep_2_index]
                        lep_2_abs_eta = abs(tree.LepEta[lep_2_index])
                        lep_2_teff = self.fake_ratios[get_FR_type(tree, lep_2_index)]
                        #print 'Looking up ', get_FR_type(tree, lep_2_index), '  eta ', lep_2_abs_eta, ' pt ', lep_2_pt
                        if lep_2_pt > self.pt_fr_max : lep_2_pt = self.pt_fr_max - 0.5
                        fr_2 = lep_2_teff.GetEfficiency(lep_2_teff.FindFixBin(lep_2_pt, lep_2_abs_eta))
                        #print 'Fake ratio: ', fr_2

                    if pass_2P2F and not skip_CR :
                        control_regions['2P2Fw'] = fr_1 / (1. - fr_1) * fr_2 / (1. - fr_2)
                        control_regions['3P1F_from_2F'] = (fr_1 / (1. - fr_1) + fr_2 / (1. - fr_2))
#                        regions['HZZ_2P2F'] = total_weight * fr_1 / (1. - fr_1) * fr_2 / (1. - fr_2)

#                        regions['HZZ_3P1F_from_2F'] = total_weight * (fr_1 / (1. - fr_1) + fr_2 / (1. - fr_2)) 
#                        if passfZMass :
#                            regions['2P2F'] = total_weight * fr_1 / (1. - fr_1) * fr_2 / (1. - fr_2)
#                            regions['3P1F_from_2F'] = total_weight * (fr_1 / (1. - fr_1) + fr_2 / (1. - fr_2)) 
 
                    if pass_3P1F and not skip_CR :
                        control_regions['3P1Fw'] = fr_1 / (1. - fr_1)
#                        regions['HZZ_3P1F'] = total_weight * fr_1 / (1. - fr_1)
#                        if passfZMass :
#                            regions['3P1F'] = total_weight * fr_1 / (1. - fr_1)

            print 'regions', regions
            to_plot = dict(regions)
            print 'selection ', regions
            if self.mode in ['BKG', 'CRZLL'] :
                print 'control_region ', control_regions 
                for selection in regions :
                    for cr in control_regions :
                        to_plot[selection + '_' + cr] = regions[selection] * control_regions[cr]
            print to_plot

            for region, final_weight in to_plot.iteritems() :                
#                print 'Filling ', region + '_ZZMass_' + final_state
                print region[:3]
                if region[:3] in  ['VBS'] and not self.isMC : continue
                self.plots[region + '_ZZMass_' + final_state].Fill(tree.ZZMass, final_weight) 
                self.plots[region + '_ZZMass_all'].Fill(tree.ZZMass, final_weight) 
                self.plots[region + '_Z1Mass_' + final_state].Fill(tree.Z1Mass, final_weight) 
                self.plots[region + '_Z1Mass_all'].Fill(tree.Z1Mass, final_weight) 
                self.plots[region + '_Z2Mass_' + final_state].Fill(tree.Z2Mass, final_weight) 
                self.plots[region + '_Z2Mass_all'].Fill(tree.Z2Mass, final_weight) 



                self.plots[region + '_Nvtx'].Fill(tree.Nvtx, final_weight) 
                self.plots[region + '_costhetastar'].Fill(tree.costhetastar, final_weight) 
                self.plots[region + '_helphi'].Fill(tree.helphi, final_weight)
                self.plots[region + '_helcosthetaZ1'].Fill(tree.helcosthetaZ1, final_weight)
                self.plots[region + '_helcosthetaZ2'].Fill(tree.helcosthetaZ2, final_weight)
                self.plots[region + '_phistarZ1'].Fill(tree.phistarZ1, final_weight)
                #print 'filling' , region + '_Njets_4p7'
                self.plots[region + '_Njets_4p7'].Fill(tree.nCleanedJetsPt30, final_weight)
                n_jets = tree.nCleanedJetsPt30
                if self.selection == 'RSE' :
                    n_jets = get_N_jets(tree) 
                self.plots[region + '_Njets_4p7_cleaned'].Fill(n_jets, final_weight)

                self.plots[region + '_Njets_4p7_int'].Fill(tree.nCleanedJetsPt30 if tree.nCleanedJetsPt30 < 5 else 4, final_weight)

                n_jets_2p4 = 0
                for i in range(len(tree.JetEta)) :
                    if abs(tree.JetEta[i]) <= 2.4 and tree.JetPt[i] > 30. :
                        n_jets_2p4 += 1
#                n_jets_2p4 = len(filter(lambda jet_eta : abs(jet_eta) < 2.4 and , tree.JetEta)) 
                if tree.nCleanedJetsPt30 == 0 and n_jets_2p4 != 0 :
                    print n_jets_2p4
                    for i in range(len(tree.JetEta)) :
                        print 'jet eta ', tree.JetEta[i], ' pt ', tree.JetPt[i]
        
                self.plots[region + '_Njets_2p4'].Fill(n_jets_2p4, final_weight)
              
                if tree.nCleanedJetsPt30 >= 1 :
                    self.plots[region + '_jet_1_pt'].Fill(tree.JetPt[0] if tree.JetPt[0] < 500 else 499., final_weight)
                    self.plots[region + '_jet_1_eta'].Fill(tree.JetEta[0], final_weight)
                    self.plots[region + '_jet_1_abseta'].Fill(abs(tree.JetEta[0]), final_weight)
    
                if tree.nCleanedJetsPt30 >= 2 :
                    self.plots[region + '_jet_2_pt'].Fill(tree.JetPt[1] if tree.JetPt[1] < 500 else 499., final_weight)
                    self.plots[region + '_jet_2_eta'].Fill(tree.JetEta[1], final_weight)
                    self.plots[region + '_jet_2_abseta'].Fill(abs(tree.JetEta[0]), final_weight)
    
                    tj_leading = 0
                    tj_subleading = 1
                    tj_eta_sum = (tree.JetEta[tj_leading] + tree.JetEta[tj_subleading]) / 2.
    
                    Z1_p4 = get_p4(tree, 0) + get_p4(tree, 1)
                    Z2_p4 = get_p4(tree, 2) + get_p4(tree, 3)
                    Z1_zepp = (Z1_p4.Eta() - tj_eta_sum)
                    Z2_zepp = (Z2_p4.Eta() - tj_eta_sum)
                    tj1_p4 = get_p4(tree, 0, do_jet = True)
                    tj2_p4 = get_p4(tree, 1, do_jet = True)
    
                    rel_pt_hard = (Z1_p4.Vect() + Z2_p4.Vect() + tj1_p4.Vect() + tj2_p4.Vect()).Pt() / (Z1_p4.Pt() + Z1_p4.Pt() + tj1_p4.Pt() + tj2_p4.Pt())
    
                    tj_delta_rel = (tj1_p4.Vect() + tj2_p4.Vect()).Pt()/ (tj1_p4.Pt() + tj2_p4.Pt())
    
                    self.plots[region + '_Z_1_zepp'].Fill(Z1_zepp, final_weight)
                    self.plots[region + '_Z_2_zepp'].Fill(Z2_zepp, final_weight) 
                    self.plots[region + '_rel_pt_hard'].Fill(rel_pt_hard, final_weight)
                    self.plots[region + '_delta_rel'].Fill(tj_delta_rel, final_weight)

                    self.plots[region + '_DiJetMass_4p7'].Fill(tree.DiJetMass, final_weight)
                    if self.isMC or self.mode != 'ZZ' or abs(tree.DiJetDEta) < 3. :
                        self.plots[region + '_DiJetDEta_4p7'].Fill(abs(tree.DiJetDEta), final_weight)
#                    self.plots[region + '_DiJetMass_PFMET'].Fill(tree.DiJetMass, tree.PFMET, final_weight)

                    #if self.isMC or self.mode != 'ZZ' or tree.ZZjj_MVA < -0.05  :
                    #    self.plots[region + '_ZZjj_MVA'].Fill(tree.ZZjj_MVA, final_weight)
                    #    self.plots[region + '_ZZjj_MVA_fine'].Fill(tree.ZZjj_MVA, final_weight)
                    #if tree.ZZjj_MVA < -0.05  :
                    #    self.plots[region + '_ZZjj_MVA_blind'].Fill(tree.ZZjj_MVA, final_weight)


                    sklearn_score = -1.
                    if self.est is not None :
                        X_test = np.array([tree.DiJetMass, abs(tree.DiJetDEta), tree.ZZMass, Z1_zepp, Z2_zepp, rel_pt_hard, tj_delta_rel])
                        #print 'X_test ', X_test
                        X_test = X_test.reshape(1, -1)
                        #print 'new X_test ', X_test
#                        sklearn_score = self.est.predict_proba(X_test)[:, 1]
                        sklearn_score = self.est.decision_function(X_test)
                        sklearn_proba = self.est.predict_proba(X_test)[:, 1]

                        sklearn_score = 2.0/(1.0+np.exp(-2.0*sklearn_score))-1
                    if self.isMC or self.mode in ['CRZLL', 'BKG'] or sklearn_score < 0.5 :
                        self.plots[region + '_sklearn_MVA'].Fill(sklearn_score, final_weight)
                        self.plots[region + '_sklearn_MVA_fine'].Fill(sklearn_score, final_weight)
                        self.plots[region + '_sklearn_MVA_SR'].Fill(sklearn_score, final_weight)
                        self.plots[region + '_sklearn_MVA_proba'].Fill(sklearn_proba, final_weight)


                    if self.create_out_tree and region == 'BLS' :
                        self.EventNumber[0] = tree.EventNumber
                        self.final_weight[0] = final_weight
                        self.m_4l[0] = tree.ZZMass
                        self.m_jj[0] = tree.DiJetMass
                        self.dEta_tj[0] = abs(tree.DiJetDEta)
                        self.Z1_zepp[0] = Z1_zepp 
                        self.Z2_zepp[0] = Z2_zepp
                        self.rel_pt_hard[0] = rel_pt_hard
                        self.tj_delta_rel[0] = tj_delta_rel
                        self.tj1_eta_x_tj2_eta[0] =  tree.JetEta[tj_leading] * tree.JetEta[tj_subleading]
                        self.out_tree.Fill()

                #if self.mode == 'BKG' :
                #    if pass_2P2F and not skip_CR :
                #        self.plots['BKG_2P2F_ZZMass_' + final_state].Fill(tree.ZZMass, total_weight * fr_1 / (1. - fr_1) * fr_2 / (1. - fr_2))
                #        self.plots['BKG_2P2F_ZZMass_FR_' + final_state].Fill(tree.ZZMass, fr_1 / (1. - fr_1) * fr_2 / (1. - fr_2), total_weight)
                #        self.plots['BKG_2P2F_ZZMass_FR_profile_' + final_state].Fill(tree.ZZMass, fr_1 / (1. - fr_1) * fr_2 / (1. - fr_2), total_weight)

                #        self.plots['BKG_3P1F_from_2F_ZZMass_' + final_state].Fill(tree.ZZMass, total_weight * (fr_1 / (1. - fr_1) + fr_2 / (1. - fr_2)))
                #    if pass_3P1F and not skip_CR :
                #        self.plots['BKG_3P1F_ZZMass_' + final_state].Fill(tree.ZZMass, total_weight * fr_1 / (1. - fr_1))
                #        self.plots['BKG_3P1F_ZZMass_FR_profile_' + final_state].Fill(tree.ZZMass, fr_1 / (1. - fr_1), total_weight)


        #if self.mode == 'ZpX_SS' :
        #    if test_bit(tree.CRflag, CRZLLss) : selection = "SS" 


 #       if self.mode == 'CRZLL' or self.mode == 'BKG' :

 #           self.plots['ZZMass_' + final_state].Fill(tree.ZZMass, total_weight) 

 #           #selection = "none"
 #           final_state_to_plot = [CR_final_state, final_state]

 #           if pass_2P2F or pass_3P1F :
 #               for plot_fs in final_state_to_plot :
 #                   if True: #not skip_CR :
 #                       self.plots[selection + '_fZMass_' + plot_fs].Fill(failZMass, total_weight)
 #                       self.plots[selection + '_ZZMass_' + plot_fs].Fill(tree.ZZMass, total_weight)
 #                   if self.selection == 'RSE' :
 #                       if pass_ss :
 #                           if pass_SIP_gt_4 : self.plots[selection + '_ss_gt4'+ '_ZZMass_' + plot_fs].Fill(tree.ZZMass, total_weight)
 #                           else : self.plots[selection + '_ss_lt4'+ '_ZZMass_' + plot_fs].Fill(tree.ZZMass, total_weight) 
 #                       else : 
 #                           if pass_SIP_gt_4 : self.plots[selection + '_os_gt4' + '_ZZMass_' + plot_fs].Fill(tree.ZZMass, total_weight) 
 #                           else : self.plots[selection + '_os_lt4' + '_ZZMass_' + plot_fs].Fill(tree.ZZMass, total_weight) 
 #             


        if self.mode == 'CRZL' :
            leading_min_pt = 20.
            subleading_min_pt = 10.
            pass_Z_lepton_pt = False
            if ((tree.LepPt[0] >= leading_min_pt and tree.LepPt[1] >= subleading_min_pt) or (tree.LepPt[0] >= subleading_min_pt and tree.LepPt[1] >= leading_min_pt)) : pass_Z_lepton_pt = True
            pass_MZ_7 = False
            pass_MZ_10 = False
            Z_pole_mass = 91.1876
            if abs(tree.Z1Mass - Z_pole_mass) < 10. : pass_MZ_10 = True
            if abs(tree.Z1Mass - Z_pole_mass) < 7. : pass_MZ_7 = True
           
            loose_lepton_id = 'none'
            if abs(tree.LepLepId[2]) in [11, 22] : loose_lepton_id = 'e'          
            if abs(tree.LepLepId[2]) == 13 : loose_lepton_id = 'm' 

            loose_lepton_pt = tree.LepPt[2]
            loose_lepton_pass = bool(tree.LepisID[2]) and (tree.LepCombRelIsoPF[2] < 0.35)
            if self.selection == 'TLE' : loose_lepton_pass = bool(tree.LepisID[2])
            loose_lepton_abs_eta = abs(tree.LepEta[2])
            pass_MET = tree.PFMET < 25.
            pass_loose_selection = False
            
           
            pass_REG = False
            if self.selection == 'REG' :
                pass_SIP = tree.LepSIP[2] < 4.0
                pass_REG = pass_SIP

            pass_RSE = False 
            if self.selection == 'RSE' :
                pass_SIP = tree.LepSIP[2] >= 4.0
                pass_RSE = pass_SIP

            
            pass_TLE = False
            TLE_dR_Z = 999.


            if self.selection == 'TLE' :
                TLE_dR_Z = get_TLE_dR_Z(tree)
                if tree.LepPt[2] > 30. and TLE_dR_Z > 1.6 : pass_TLE = True 
            pass_loose_selection = (self.selection == 'REG' and pass_REG) or (self.selection == 'RSE' and pass_RSE) or (self.selection == 'TLE' and pass_TLE) 
            eta_bin = ''

            if loose_lepton_id == 'e' :
               if loose_lepton_abs_eta < 1.479 : eta_bin = 'EB'
               else : eta_bin = 'EE'

            if loose_lepton_id == 'm' :
                if loose_lepton_abs_eta < 1.2 : eta_bin = 'EB'
                else : eta_bin = 'EE'


            final_states = []
            final_states.append('Zp' + loose_lepton_id)
            if abs(tree.LepLepId[0]) == abs(tree.LepLepId[1]) == 11 : final_states.append('2ep' + loose_lepton_id)
            if abs(tree.LepLepId[0]) == abs(tree.LepLepId[1]) == 13 : final_states.append('2mp' + loose_lepton_id)
            if abs(tree.LepLepId[0]) != abs(tree.LepLepId[1]) :
               sys.exit("ERROR: Z candiate in CR is not same flavor!") 

            if loose_lepton_pt > self.pt_fr_max : loose_lepton_pt = self.pt_fr_max - 0.0001

            if pass_loose_selection :
                for fs in final_states :
                    self.plots[fs + '_Z1_mass_all_' + eta_bin].Fill(tree.Z1Mass, total_weight)
                    self.plots[fs + '_event_ETmiss_all_' + eta_bin].Fill(tree.PFMET, total_weight)
   
                    if pass_Z_lepton_pt and pass_MET and pass_MZ_7 :
                        self.plots[fs + '_ptl3_all_' + eta_bin].Fill(loose_lepton_pt, total_weight) 
                        self.plots[fs + '_ptl3_all'].Fill(loose_lepton_pt, loose_lepton_abs_eta, total_weight)

                    if pass_loose_selection : 
                        if loose_lepton_pass and pass_Z_lepton_pt and pass_MET : 
                            self.plots[fs + '_Z1_mass_pass_' + eta_bin].Fill(tree.Z1Mass, total_weight) 
                     
                        if loose_lepton_pass and pass_Z_lepton_pt and pass_MZ_7 :
                            self.plots[fs + '_event_ETmiss_pass_' + eta_bin].Fill(tree.PFMET, total_weight)

                        if loose_lepton_pass and pass_Z_lepton_pt and pass_MET and pass_MZ_7 :
                            self.plots[fs + '_ptl3_pass_' + eta_bin].Fill(loose_lepton_pt, total_weight)
                            self.plots[fs + '_ptl3_pass'].Fill(loose_lepton_pt, loose_lepton_abs_eta, total_weight)
         
            if pass_MZ_7 and pass_Z_lepton_pt :
                if self.selection == 'TLE' and loose_lepton_id == 'e' and tree.LepPt[2] > 30. :
                    self.plots['Zpe' + '_TLE_dR_Z_' + eta_bin].Fill(TLE_dR_Z, total_weight)
                if pass_loose_selection :
                       self.plots['Zp' + loose_lepton_id + '_NoMET_fakeRatio_' + eta_bin].Fill(loose_lepton_pass, loose_lepton_pt)
                       if pass_MET :
                           self.plots['Zp' + loose_lepton_id + '_fakeRatio_' + eta_bin].Fill(loose_lepton_pass, loose_lepton_pt)
                           self.plots['Zp' + loose_lepton_id +  '_fakeRatio'].Fill(loose_lepton_pass, loose_lepton_pt, loose_lepton_abs_eta)
                           
                if pass_MET and self.selection == 'REG' and pass_SIP == False : self.plots['Zp' + loose_lepton_id +  '_fakeRatio_from_REG_' + eta_bin].Fill(loose_lepton_pass, loose_lepton_pt)
                if pass_MET and self.selection == 'REG' and pass_SIP == False : self.plots['Zp' + loose_lepton_id +  '_fakeRatio_from_REG'].Fill(loose_lepton_pass, loose_lepton_pt, loose_lepton_abs_eta)

                lep_1 = ROOT.TLorentzVector()
                lep_1.SetPtEtaPhiM(tree.LepPt[0], tree.LepEta[0], tree.LepPhi[0], 0)

                lep_2 = ROOT.TLorentzVector()
                lep_2.SetPtEtaPhiM(tree.LepPt[1], tree.LepEta[1], tree.LepPhi[1], 0)

                lep_3 = ROOT.TLorentzVector()
                lep_3.SetPtEtaPhiM(tree.LepPt[2], tree.LepEta[2], tree.LepPhi[2], 0)

                pass_m2l = False
                if (lep_1 + lep_3).M() > 4. and (lep_2 + lep_3).M() > 4. : pass_m2l = True 
                if pass_MET and self.selection == 'REG' and pass_SIP == False and pass_m2l == True : self.plots['Zp' + loose_lepton_id +  '_fakeRatio_from_REG_DoubleM2l_' + eta_bin].Fill(loose_lepton_pass, loose_lepton_pt)


                if pass_Z_lepton_pt :
                    if pass_loose_selection :
                       if pass_MET :
                           self.plots['Zp' + loose_lepton_id + '_fakeRatio_' + eta_bin + '_mZ1'].Fill(loose_lepton_pass, tree.Z1Mass)
            

        return 0

    def SlaveTerminate(self):
        #print 'py: slave terminating'
        return
        

    def Terminate(self):
        #print '!!!!Terminate'
        
        f = ROOT.TFile(self.outputName, 'recreate')
#        with root_open(self.outputName,
#                       'recreate') as f:
            # write original histograms:

#        if 'is_skim' in self.tree_options['booleans'] :
#            pre_skim_n_events = self.tree_options['pre_skim_n_events']
#            self.plots['skim_n_events'].SetBinContent(1, pre_skim_n_events)


        selections = []
        #if self.mode == 'CRZLL' or self.mode == 'BKG' :
        #    selections = ['2P2F', '3P1F']
        #if self.mode == 'BKG' :
        #    selections += ['BKG']
        #if self.mode == 'ZZ' :
        selections += ['BLS', 'ZZ','HZZ', 'VBS', 'nVBS']
        for sel in selections :
            f.mkdir(sel)

        f.mkdir('raw')
        f.raw.cd()
        #f.raw.cd()
        #weight_plot = None
        #for l in self.GetOutputList() :
        #    if l.GetName() == 'weights' :
        #        weight_plot = l
        #        break
        # 
        #sum_weights = weight_plot.GetBinContent(weight_plot.FindBin(1))
        #print 'Sum of weights: ', sum_weights 
        #if sum_weights == 0. :
        #    sum_weights = 1.
        out_tree = None
        for l in self.GetOutputList():
         
            #print 'Object: ', l.GetName() 
            object_name = l.GetName()
            if isinstance(l, ROOT.TTree) :
                print 'Found tree'
                out_tree = l
                print out_tree
                continue
            folder_name = 'raw' 
            for sel in selections :
                if object_name.find(sel) == 0 :
                    folder_name = sel
            #try :
                #scale = getattr(l, 'Scale')
                #print scale
                #if not isinstance(l, TProfile) :
                    #scale(1. / sum_weights)
                #print 'scaled!'
            #except AttributeError :
            #    print 'Did not scale ', l.GetName() 
            #print name
            #prefix = name[:3]
            #print prefix
            #f.cd(prefix)
            f.cd(folder_name)
    
            l.Write()
        f.Close()
        print 'Successfully wrote results to ', self.outputName

        if self.create_out_tree :
            for l in self.GetOutputList():
                object_name = l.GetName()
                if isinstance(l, ROOT.TTree) :
                    print 'Writing output tree in file ', 'out_tree_' + self.outputName
                    f = ROOT.TFile.Open('out_tree_' + self.outputName, "RECREATE")
                    l.Write()
                    f.Close()
    def Init(self, tree):
        self.fChain.SetBranchStatus('*', 1)
