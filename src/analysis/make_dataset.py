import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import pyvista as pv
import glob
import os

print("🚀 데이터 결합 파이프라인 시작...\n")

# 🌟 [현업 꿀팁] 이 파이썬 파일이 있는 '진짜 위치(작은방)'를 스스로 찾게 만듭니다!
# (이렇게 하면 재생 버튼을 누르든, 터미널에서 치든 절대 길을 잃지 않습니다.)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("1. 남해안 정답지 데이터를 불러오고 가상 관측망(반경 2km)을 생성합니다.")

# 🌟 정원님께서 'final_data' 폴더를 어디에 만드셨는지에 따라 경로를 찾아갑니다.
# (현재 코드는 yeosu/final_data 에 있다고 가정하고 세팅되었습니다.)
csv_path = os.path.join(BASE_DIR, '../../final_data/namhae_wide_marine_debris_windows (1).csv')

# 만약 final_data 폴더가 yeosu/src/analysis/final_data 안에 있다면
# 바로 윗줄 맨 앞에 #을 붙이고, 아래 줄의 #을 지우시면 됩니다!
# csv_path = os.path.join(BASE_DIR, 'final_data/namhae_wide_marine_debris_windows (1).csv')

if not os.path.exists(csv_path):
    print(f"❌ 에러: '{csv_path}' 파일을 찾을 수 없습니다! final_data 폴더 위치를 확인해주세요.")
    exit()

df_truth = pd.read_csv(csv_path)
geometry = [Point(xy) for xy in zip(df_truth['Longitude'], df_truth['Latitude'])]
gdf_truth = gpd.GeoDataFrame(df_truth, geometry=geometry, crs="EPSG:4326")
gdf_truth_buffers = gdf_truth.to_crs("EPSG:5179").buffer(2000).to_crs("EPSG:4326")
gdf_truth['geometry'] = gdf_truth_buffers 
gdf_truth['mohid_particle_count'] = 0 

# 2. MOHID 물리 데이터(VTU) 불러오기 (예제 데이터는 yeosu/example 에 있죠!)
vtu_folder = os.path.join(BASE_DIR, '../../example')
vtu_files = sorted(glob.glob(f"{vtu_folder}/*.vtu"))

if len(vtu_files) == 0:
    print(f"❌ 에러: '{vtu_folder}' 폴더 안에 .vtu 파일이 없습니다!")
    exit()

print(f"2. 총 {len(vtu_files)}개의 MOHID 시간대 데이터를 분석합니다... (조금 걸릴 수 있습니다 ☕️)")

for file_path in vtu_files:
    mesh = pv.read(file_path)
    points = mesh.points
    df_step = pd.DataFrame({'lon': points[:, 0], 'lat': points[:, 1]})
    step_geom = [Point(xy) for xy in zip(df_step['lon'], df_step['lat'])]
    gdf_step = gpd.GeoDataFrame(df_step, geometry=step_geom, crs="EPSG:4326")
    
    # 공간 결합(Spatial Join)
    matched = gpd.sjoin(gdf_step, gdf_truth, how="inner", predicate="within")
    step_counts = matched['지역명'].value_counts()
    
    for region, count in step_counts.items():
        gdf_truth.loc[gdf_truth['지역명'] == region, 'mohid_particle_count'] += count

# 3. 결과 확인 및 저장
print("\n✅ 분석 완료! AI가 학습할 '문제집' 결과입니다:")
final_dataset = gdf_truth[['지역명', 'mohid_particle_count', '수량(개)', '무게(kg)']]
print(final_dataset.head(10))

# 완성된 결과물도 깔끔하게 정답지가 있는 final_data 폴더 안에 같이 저장합니다!
save_path = os.path.join(os.path.dirname(csv_path), "AI_Training_Dataset.csv")
final_dataset.to_csv(save_path, index=False, encoding='utf-8-sig')
print(f"\n💾 '{save_path}' 파일로 저장이 완료되었습니다!")