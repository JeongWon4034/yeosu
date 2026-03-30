import geopandas as gpd

# 1. 남해안 쉐이프파일 세트 불러오기
print("데이터 읽는 중...")
gdf = gpd.read_file("namhae_final3.shp", encoding='utf-8')

# 2. 대시보드를 가볍게 만들기 위해 '여수' 섬만 필터링 (선택 사항)
# name 컬럼에 '여수'가 들어간 행만 추출
gdf_yeosu = gdf[gdf['name'].str.contains("여수", na=False)]

# 3. 하나의 GeoJSON 파일로 변환하여 저장
gdf_yeosu.to_file("yeosu_polygons.geojson", driver="GeoJSON")
print("변환 완료! 폴더에 yeosu_polygons.geojson 파일이 생겼습니다!")