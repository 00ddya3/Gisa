import pandas as pd
import warnings

warnings.filterwarnings(action='ignore')
pd.set_option('display.max_column', 20)

df = pd.read_csv('gisa/bigData-main/youtube.csv')

#인기동영상 제작횟수가 많은 채널 상위 10개명
df1 = df.groupby('channelId').count()
df1 = df1.sort_values('video_id', ascending=False)
df1 = df1.head(10).index.to_list()
lst1 = df.loc[df['channelId'].isin(df1), 'channelTitle'].unique()
lst1 = list(lst1)
print(lst1)

#dislike 수가 like 수보다 높은 동영상을 제작한 채널
df2 = df.loc[df['dislikes'] > df['likes'], 'channelTitle'].unique()
lst2 = list(df2)
print(lst2)

#채널명을 바꾼 채널의 갯수
df3 = df[['channelId', 'channelTitle']]
df3 = df3.drop_duplicates()
df3 = df3.groupby(by='channelId').count()
df3 = df3[df3['channelTitle'] > 1]
print(df3.shape[0])

#일요일에 인기있었던 영상 중 가장 많은 영상 카테고리
df4 = df[['categoryId', 'trending_date']]
df4['trending_date'] = pd.to_datetime(df4['trending_date'])
df4['weekday'] = df4['trending_date'].dt.day_name()
df4 = df4[df4['weekday'] == 'Sunday']
df4 = df4['categoryId'].value_counts()
print(df4.index[0])

#각 요일별 인기 영상들의 categoryid는 각각 몇개씩인지 데이터프레임으로 ~~
df5 = df[['categoryId', 'trending_date']]
df5['trending_date'] = pd.to_datetime(df5['trending_date'])
df5['weekday'] = df5['trending_date'].dt.day_name()
df5 = df5.groupby(by=['weekday','categoryId'], as_index=False).count()
df5 = df5.pivot(index = 'categoryId', columns='weekday')
print(df5)

#viewcount 대비 댓글수가 가장 높은 영상
df6 = df[['title', 'view_count', 'comment_count']]
df6.drop(df6[df6['view_count'] == 0].index, inplace=True)
df6['ratio'] = df6['comment_count'] / df6['view_count']
df6 = df6.sort_values('ratio', ascending=False)
print(df6['title'].head(1))

#viewcount 대비 댓글수가 가장 낮은 영상
df7 = df[['title', 'view_count', 'comment_count']]
df7.drop(df7[df7['view_count'] == 0].index, inplace=True)
df7.drop(df7[df7['comment_count'] == 0].index, inplace=True)
df7['ratio'] = df7['comment_count'] / df7['view_count']
df7 = df7.sort_values('ratio', ascending=True)
print(df7['title'].head(1))

#like 대비 dislike 수가 가장 적은 영상
df8 = df[['title', 'likes', 'dislikes']]
df8.drop(df8[df8['likes'] == 0].index, inplace=True)
df8.drop(df8[df8['dislikes'] == 0].index, inplace=True)
df8['ratio'] = df8['dislikes'] / df8['likes']
df8 = df8.sort_values('ratio', ascending=True)
print(df8['title'].head(1).to_list()[0])

#가장 많은 트렌드 영상을 제작한 채널명
df9 = df['channelTitle'].value_counts()
print(df9.index[0])

#20일 이상 인기동영상 리스트에 포함된 동영상의 수
df10 = df['video_id'].value_counts()
df10 = df10[df10 >= 20]
print(df10.shape[0])
