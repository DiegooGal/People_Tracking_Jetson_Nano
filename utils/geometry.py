#=================================================#
# geometry.py - v2                                #
# Funciones geométricas para tracking y detección #
# 17/01/2026                                      #
#=================================================#

import numpy as np
from numba import jit


@jit(nopython=True)
def iou(bb_test, bb_gt):
    #================================================================#
    # Calcula Intersection over Union (IoU) entre dos bounding boxes #
    #================================================================#
    
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
              + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
    return o


def convert_bbox_to_z(bbox):
    #=====================================================#
    # Convierte bounding box de [x1,y1,x2,y2] a [u,v,s,r] #
    #=====================================================#
    
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h  # área
    r = w / float(h + 1e-6)  # aspect ratio
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    #=======================================================#
    # Convierte de formato centro [x,y,s,r] a [x1,y1,x2,y2] #
    #=======================================================#
    
    # Validar valores positivos
    if x[2] <= 0 or x[3] <= 0:
        if score is None:
            return np.array([0, 0, 0, 0]).reshape((1, 4))
        else:
            return np.array([0, 0, 0, 0, score]).reshape((1, 5))
    
    w = np.sqrt(x[2] * x[3])
    h = x[2] / (w + 1e-6)
    
    if score is None:
        return np.array([x[0] - w / 2., x[1] - h / 2., 
                        x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., 
                        x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    #=====================================================#
    # Asigna detecciones a trackers existentes usando IoU #
    #=====================================================#
    
    from sklearn.utils.linear_assignment_ import linear_assignment
    
    if len(trackers) == 0:
        return (np.empty((0, 2), dtype=int), 
                np.arange(len(detections)), 
                np.empty((0, 5), dtype=int))
    
    # Calcular matriz de IoU
    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
    
    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det, trk)
    
    # Algoritmo Húngaro para asignación óptima
    matched_indices = linear_assignment(-iou_matrix)
    
    # Identificar detecciones y trackers no emparejados
    unmatched_detections = []
    for d in range(len(detections)):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
            
    unmatched_trackers = []
    for t in range(len(trackers)):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)
    
    # Filtrar matches con IoU bajo
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)
    
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
