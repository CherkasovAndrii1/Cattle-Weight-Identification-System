// src/constants/tabNames.js
export const TAB_KEYS = {
    SEGMENTATION: 'segmentation',
    KEYPOINTS: 'keypoints',
    WEIGHT: 'weight',
};

export const TAB_NAMES = {
    [TAB_KEYS.SEGMENTATION]: 'Segmentation',
    [TAB_KEYS.KEYPOINTS]: 'Keypoint extraction',
    [TAB_KEYS.WEIGHT]: 'Weight estimation',
};

export const TABS_CONFIG = [
    { key: TAB_KEYS.SEGMENTATION, name: TAB_NAMES[TAB_KEYS.SEGMENTATION] },
    { key: TAB_KEYS.KEYPOINTS, name: TAB_NAMES[TAB_KEYS.KEYPOINTS] },
    { key: TAB_KEYS.WEIGHT, name: TAB_NAMES[TAB_KEYS.WEIGHT] },
];