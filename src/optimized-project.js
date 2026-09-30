import {validateTracking,trackingVideoCompatible} from './tracking-data.js';
import {networkGeometryKey} from './match-network.js';

// Refinement JSON keeps the complete tracking state. The source calibration ZIP
// supplies the unchanged lens maps and brightness field.
export function importOptimizedProject(value,calibration,video=null) {
  if(value?.format!=='rasterlabor-pose-refinement'||value.version!==1)
    throw new Error('Unbekanntes Optimierungsformat.');
  if(!calibration?.maps)throw new Error('Zuerst das zugehoerige Kalibrierungs-ZIP laden.');
  if(value.geometryKey!==networkGeometryKey(calibration))
    throw new Error('Die Optimierung gehoert zu einer anderen Linsenkalibrierung.');
  const tracking=validateTracking(value.tracking);
  if(!tracking?.path?.length)throw new Error('Die Optimierung enthaelt keine Frames.');
  if(video&&!trackingVideoCompatible(tracking,video))throw new Error('Video und Optimierung passen nicht zusammen.');
  return tracking;
}
