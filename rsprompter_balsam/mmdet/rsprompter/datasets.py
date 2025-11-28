# NOTICE: This file is a modification of a file from the 
#  RSPrompter project, which is licensed under Apache 2.0.
# The original file can be found at: https://github.com/KyanChen/RSPrompter/blob/release/mmdet/rsprompter/datasets.py
#
# All modifications are Copyright (c) 2025 Mélisande Teng.
# Original Copyright (c) 2018-2023 RSPrompter. 
# -------------------------------------------------------------------------


from mmdet.datasets import CocoDataset, CocoDSMDataset
from mmdet.registry import DATASETS


@DATASETS.register_module()
class NWPUInsSegDataset(CocoDataset):
    METAINFO = {
        'classes': ['airplane', 'ship', 'storage_tank', 'baseball_diamond',
                    'tennis_court', 'basketball_court', 'ground_track_field',
                    'harbor', 'bridge', 'vehicle'],
        'palette': [(220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
                    (0, 60, 100), (0, 80, 100), (0, 0, 230),
                    (119, 11, 32), (0, 255, 0), (0, 0, 255)]
    }


@DATASETS.register_module()
class WHUInsSegDataset(CocoDataset):
    METAINFO = {
        'classes': ['building'],
        'palette': [(0, 255, 0)]
    }


@DATASETS.register_module()
class SSDDInsSegDataset(CocoDataset):
    METAINFO = {
        'classes': ['ship'],
        'palette': [(0, 0, 255)]
    }
    
@DATASETS.register_module()
class TreesInsSegDataset(CocoDataset):
    METAINFO = {
        'classes': ['piba','pima', 'pist','pigl', 'thoc', 'ulam', 'other', 'beal', 'acsa'],
        'palette': [(220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),  (0, 60, 100), (0, 80, 100), (0, 0, 230),
                    (119, 11, 32), (0, 255, 0)]
    }

@DATASETS.register_module()   
class TreesInsSegSBLDataset(CocoDataset):
    METAINFO = {
        'classes': ['dead', 'Pinopsida', 'Magnoliopsida', 'Thuja occidentalis L.','Abies balsamea (L.) Mill.','Larix laricina (Du Roi) K.Koch','Tsuga canadensis (L.)','Betula L.','Fagus grandifolia Ehrh.','Populus L.','Acer L.','Acer pensylvanicum L.','Acer saccharum Marshall','Acer rubrum L.','Pinus strobus L.','Betula alleghaniensis Britton','Betula papyrifera Marshall','Picea A.Dietr.'],
        'palette': [(220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),  (0, 60, 100), (0, 80, 100), (0, 0, 230),
                    (119, 11, 32), (0, 255, 0), (0, 0, 255), (0,60,60), (60,60,0), (60,0,60), (100,80,0), (20,60,200), (60,200,20), (11,32,119), (32,119,11)]
    }    
    
    
@DATASETS.register_module()   
class TreesInsSegBCIDataset(CocoDataset):
    METAINFO = {
        'classes': ['Burseraceae','Fabaceae','Arecaceae','Rutaceae','Malvaceae','Euphorbiaceae','Bignoniaceae','Annonaceae', 'Urticaceae','Rubiaceae', 'Myristicaceae', 'Apocynaceae', 'Cordiaceae', 'Meliaceae', 'Other', 'Simaroubaceae', 'Moraceae', 'Anacardiaceae', 'Sapotaceae', 'Nyctaginaceae', 'Lauraceae', 'Lecythidaceae', 'Phyllanthaceae', 'Araliaceae', 'Salicaceae','Chrysobalanaceae','Cannabaceae', 'Putranjivaceae', 'Combretaceae', 'Melastomataceae', 'Calophyllaceae'],
        'palette': [  (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
    (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
    (192, 0, 0), (0, 192, 0), (0, 0, 192), (192, 192, 0),
    (192, 0, 192), (0, 192, 192), (64, 0, 0), (0, 64, 0),
    (0, 0, 64), (64, 64, 0), (64, 0, 64), (0, 64, 64),
    (255, 128, 0), (255, 0, 128), (128, 255, 0), (0, 255, 128),
    (128, 0, 255), (0, 128, 255), (128, 128, 128)]
    }    
    
@DATASETS.register_module()   
class TreesInsSegDSMSBLDataset(CocoDSMDataset):
    METAINFO = {
        'classes': ['dead', 'Pinopsida', 'Magnoliopsida', 'Thuja occidentalis L.','Abies balsamea (L.) Mill.','Larix laricina (Du Roi) K.Koch','Tsuga canadensis (L.)','Betula L.','Fagus grandifolia Ehrh.','Populus L.','Acer L.','Acer pensylvanicum L.','Acer saccharum Marshall','Acer rubrum L.','Pinus strobus L.','Betula alleghaniensis Britton','Betula papyrifera Marshall','Picea A.Dietr.'],
        'palette': [(220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),  (0, 60, 100), (0, 80, 100), (0, 0, 230),
                    (119, 11, 32), (0, 255, 0), (0, 0, 255), (0,60,60), (60,60,0), (60,0,60), (100,80,0), (20,60,200), (60,200,20), (11,32,119), (32,119,11)]
    }    
    

@DATASETS.register_module()
class TreesInsSegDSMDataset(CocoDSMDataset):
    METAINFO = {
        'classes': ['piba','pima', 'pist','pigl', 'thoc', 'ulam', 'other', 'beal', 'acsa'],
        'palette': [(220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),  (0, 60, 100), (0, 80, 100), (0, 0, 230),
                    (119, 11, 32), (0, 255, 0)]
    }

@DATASETS.register_module()
class TreesInsSegDSMBCIDataset(CocoDSMDataset):
    METAINFO = {
        'classes': ['Burseraceae','Fabaceae','Arecaceae','Rutaceae','Malvaceae','Euphorbiaceae','Bignoniaceae','Annonaceae', 'Urticaceae','Rubiaceae', 'Myristicaceae', 'Apocynaceae', 'Cordiaceae', 'Meliaceae', 'Other', 'Simaroubaceae', 'Moraceae', 'Anacardiaceae', 'Sapotaceae', 'Nyctaginaceae', 'Lauraceae', 'Lecythidaceae', 'Phyllanthaceae', 'Araliaceae', 'Salicaceae','Chrysobalanaceae','Cannabaceae', 'Putranjivaceae', 'Combretaceae', 'Melastomataceae', 'Calophyllaceae'],
        'palette': [  (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
    (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
    (192, 0, 0), (0, 192, 0), (0, 0, 192), (192, 192, 0),
    (192, 0, 192), (0, 192, 192), (64, 0, 0), (0, 64, 0),
    (0, 0, 64), (64, 64, 0), (64, 0, 64), (0, 64, 64),
    (255, 128, 0), (255, 0, 128), (128, 255, 0), (0, 255, 128),
    (128, 0, 255), (0, 128, 255), (128, 128, 128)]
    }    
    