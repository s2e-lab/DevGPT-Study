import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut

# DICOM 파일 로드
dcm = pydicom.dcmread('파일 경로')

# 'Modality' lookup table 적용
data = apply_modality_lut(dcm.pixel_array, dcm)

# 'VOI' lookup table 적용
data = apply_voi_lut(data, dcm)

# 결과 확인
print(data)
