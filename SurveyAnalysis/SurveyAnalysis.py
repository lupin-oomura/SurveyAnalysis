import pandas as pd
import json
import re
import jaconv #半角→全角用
import queue
import os

from dotenv import load_dotenv
from openai import OpenAI

# クラスタリング用
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer #TF-IDF用

#wordcloud用
import MeCab
from collections import Counter
from wordcloud import WordCloud

class SurveyAnalysis :

    df_setting      = None  # セッティングデータ(項目=QID,質問,質問種類,入力種類,分類,ワードクラウド,平均点,示唆,ネガ抽出,要望抽出)
    df_surveyresult = None  # サーベイ結果データ
    client          = None  # for OpenAI
    df_fa           = pd.DataFrame() #テキスト（フリーアンサー）に関するデータ。ベクトル値やクラスターなどが記録される。
    df_cluster_info = pd.DataFrame() #クラスターごとの情報（タイトルや代表コメントなど）
    queue_progress  = None
    gpt_model       = None
    fld_workdata    = None

    def __init__(self, fld_workdata:str='workdata') :
        load_dotenv()
        self.fld_workdata   = fld_workdata
        self.client         = OpenAI()
        self.gpt_model      = 'gpt-4o'
        self.queue_progress = queue.Queue()

        if not os.path.exists(self.fld_workdata):
            os.makedirs(self.fld_workdata)

        
    def set_data(self, df_setting:pd.DataFrame, df_surveyresult:pd.DataFrame) :
        self.df_setting      = df_setting
        self.df_surveyresult = df_surveyresult

    def data_kakou(self) :
        self.queue_progress.put(f'データ加工中')

        #データ加工：FA(FreeAnswer)を縦持ちにして、全角化する。
        self.df_fa  = pd.DataFrame(columns=['qid', 'original_index', 'text', 'vector', 'cluster'])
        self.df_cluster_info = pd.DataFrame(columns=['qid', 'cluster', 'title', '代表的な文章'])
        for idx, row in self.df_setting[self.df_setting['質問種類']=='FA'].iterrows() :
            df_temp = self.df_surveyresult[[row.qid]].reset_index().rename(columns={'index':'original_index', row.qid:'text'})
            df_temp = df_temp[df_temp['text'].str.strip() != ''].dropna() #空行削除
            df_temp['text'] = df_temp['text'].apply(self.__han_to_zen) #全角化
            if row['改行読点で切る？'] == '〇' :
                l_texts = []
                l_index = []
                # Seriesの各要素に対して処理を行う
                for idx, text in zip(df_temp['original_index'], df_temp['text']):
                    split_parts = re.split(r'。|\n', text)
                    for part in split_parts:
                        if part.strip():  # 空文字列を無視
                            l_texts.append(part.strip() + "。")  # 「。」を追加して整形
                            l_index.append(idx)
                df_temp = pd.DataFrame({'original_index': l_index, 'text': l_texts})

            if row['分類'] == "〇" :
                #---------------------------------------------------#
                #--- ベクトル化 -------------------------------------#
                #---------------------------------------------------#
                self.queue_progress.put(f'({row.qid}) 文章のベクトル化処理中...')

                if row['ベクトル種類'] == 'embedding' :
                    if pd.notna(row['emb-model']) :
                        df_temp['vector'] = self.__myembedding(df_temp['text'], row['emb-model'])
                    else :
                        df_temp['vector'] = self.__myembedding(df_temp['text'])
                elif row['ベクトル種類'] == 'tfidf' :
                    if pd.notna(row['tfidf対象品詞']) :
                        hinshi = row['tfidf対象品詞'].split(',')
                        hinshi = [x.strip() for x in hinshi]
                        df_temp['vector'] = self.__mytfidf(df_temp['text'], hinshi)
                    else :
                        df_temp['vector'] = self.__mytfidf(df_temp['text'])
                else :
                    print("ベクトル種類の設定値が異常です！！")

                #---------------------------------------------------#
                #--- クラスタ数の最適化 & 分類 -----------------------#
                #---------------------------------------------------#
                self.queue_progress.put(f'({row.qid}) クラスタリング中...')

                #最適化
                target_clustersize = row['クラスター数'] if pd.notna(row['クラスター数']) else 0
                if target_clustersize and target_clustersize > 0 :
                    pass
                else :
                    target_clustersize = self.__optimal_clustersize(df_temp["vector"], n_cluster_min=3, n_cluster_max=9)
                    print(f'Optimal number of clusters: {target_clustersize}')
                #クラスタリング
                df_temp['cluster'] = self.__calculate_cluster(df_temp["vector"].tolist(), target_clustersize)

                #---------------------------------------------------------#
                #--- クラスターのタイトル付け ------------------------------#
                #---------------------------------------------------------#
                df_cluster_title = self.__gpt_generate_title(df_temp, self.gpt_model)

                #清書
                df_cluster_title['qid'] = row.qid
                self.df_cluster_info    = pd.concat([self.df_cluster_info, df_cluster_title], axis=0)
                del df_cluster_title


            #清書
            df_temp['qid']          = row.qid
            self.df_fa              = pd.concat([self.df_fa, df_temp], axis=0)
            del df_temp

        self.df_fa           = self.df_fa          .reset_index(drop=True)
        self.df_cluster_info = self.df_cluster_info.reset_index(drop=True)

        self.savedata()
        self.queue_progress.put(f'データ加工完了')

        
    #--- 示唆出し ---------------------------------------------#
    def generate_suggest(self) :
        df_gattai = pd.DataFrame()
        for idx, row in self.df_setting[self.df_setting['質問種類']=='FA'].iterrows() :
            if row['分類'] == "〇" and row['示唆'] == '〇' :
                self.queue_progress.put(f'({row.qid}) 示唆出し処理中...')
                print(f"--- Question : {row.qid} --------")
                df_temp = self.df_fa[self.df_fa.qid == row.qid].copy()
                q = str( self.df_setting[self.df_setting.qid==row.qid]['質問'].iloc[0] )
                df_suggest = self.__gpt_generate_suggest(df_temp, q, self.gpt_model)

                #清書
                df_suggest['qid'] = row.qid
                df_gattai = pd.concat([df_gattai, df_suggest], axis=0)
                del df_suggest  

        if '示唆' in self.df_cluster_info.columns.tolist() :
            del self.df_cluster_info['示唆']
        self.df_cluster_info = self.df_cluster_info.merge(df_gattai[['qid','cluster','示唆']], on=['qid','cluster'], how='left')
        del df_gattai
        self.queue_progress.put(f'示唆出し処理完了')


    #--- Score -----------------------------------------------#
    def add_score(self) :
        self.queue_progress.put(f'スコア集計処理中...')
        df_gattai = pd.DataFrame()
        for idx, row in self.df_setting[self.df_setting['質問種類']=='FA'].iterrows() :
            if pd.notna(row['平均点']) :
                scoreq = row['平均点']
                df_temp = self.df_fa[self.df_fa.qid == row.qid].copy()
                df_score = self.df_surveyresult[scoreq].rename('score')
                df_temp  =df_temp.merge(df_score, left_on='original_index', right_index=True, how='inner')
                xxx = df_temp.groupby('cluster')['score'].mean().reset_index()
                xxx['qid'] = row.qid

                df_gattai = pd.concat([df_gattai, xxx], axis=0)
                del df_temp, df_score, xxx

        if 'score' in self.df_cluster_info.columns.tolist() :
            del self.df_cluster_info['score']
        self.df_cluster_info = self.df_cluster_info.merge(df_gattai[['qid','cluster','score']], on=['qid','cluster'], how='left')
        del df_gattai
        self.queue_progress.put(f'スコア集計処理完了')



    #--- Word Cloud ------------------------------------------#
    def generate_wordcloud(self) :
        self.queue_progress.put(f'ワードクラウド処理中...')
        for idx, row in self.df_setting[self.df_setting['f_wordcloud']==True].iterrows() :
            self.queue_progress.put(f'ワードクラウド処理中...({row.qid})')
            exclude_words = ['こと']
            l_txt = self.df_fa[self.df_fa.qid==row.qid]['text'].tolist()
            fn_wordcloud = f'wordcloud_{row.qid}.png'
            self.__myworcloud(l_txt, exclude_words, fn_wordcloud)
        self.queue_progress.put(f'ワードクラウド処理完了')

    #--- HTMLファイルを作る -------------------------------------------------#
    def generate_html(self, fld:str='templates') :
        self.queue_progress.put(f'Webページ生成中')

        if not os.path.exists(fld):
            os.makedirs(fld)

        l_q = self.df_setting[self.df_setting['分類']=='〇']['qid'].tolist()
        data = {}

        for target_q in l_q :
            result = []
            for _, row in self.df_cluster_info[self.df_cluster_info.qid==target_q].iterrows():
                item = {
                    "title": row['title'],
                    "score": round(row['score'], 2),
                    "representative_comments": [],
                    "suggestions": []    
                }
                item["representative_comments"] = row['代表的な文章'] 
                item["suggestions"            ] = row['示唆'] 

                result.append(item)


            data[target_q] = result


        txt_data = json.dumps(data, ensure_ascii=False, indent=4)

        txt_table = "                <tr><td>Q1</td><td>Id</td><td>属性</td></tr>\n"
        txt_qtext = ""
        txt_questions = ""
        for _, row in self.df_setting[self.df_setting['分類']=='〇'].iterrows() :
            txt_table += f"                <tr><td>{row.qid}</td><td>{row.質問}</td><td>{row.質問種類}</td></tr>\n"
            txt_qtext += f'        <li onclick="showResults(\'{row.qid}\')">{row.qid}</li>\n'
            txt_questions += f'"{row.qid}":"{row.質問}",\n'
        with open(f"{fld}/index_base.html", "r", encoding='utf-8') as f :
            txt_html = f.read()

        txt_html = txt_html.replace('■■■questionnaire■■■', txt_qtext) \
            .replace('■■■table■■■', txt_table) \
                .replace('■■■data■■■', txt_data) \
                    .replace('■■■questions■■■', txt_questions) \
                        .replace('"suggestions": null', '"suggestions": []')


        with open(f"{fld}/index.html", "w", encoding='utf-8') as f :
            f.write(txt_html)

        self.queue_progress.put(f'Webページ生成完了')


    #--- Save ------------------------------------------------#
    def savedata(self) :
        self.df_setting     .to_json(f'{self.fld_workdata}/df_setting.json'     , orient='records', lines=True, force_ascii=False)
        self.df_surveyresult.to_json(f'{self.fld_workdata}/df_surveyresult.json', orient='records', lines=True, force_ascii=False)
        # self.df_fa.to_parquet(f'{self.fld_workdata}/df_fa.parquet', engine='pyarrow')
        self.df_fa          .to_json(f'{self.fld_workdata}/df_fa.json'          , orient='records', lines=True, force_ascii=False)
        self.df_cluster_info.to_json(f'{self.fld_workdata}/df_cluster_info.json', orient='records', lines=True, force_ascii=False)

    #--- Load ------------------------------------------------#
    def loaddata(self) :
        #読み込み
        self.df_setting      = pd.read_json(f'{self.fld_workdata}/df_setting.json'     , orient='records', lines=True)
        self.df_surveyresult = pd.read_json(f'{self.fld_workdata}/df_surveyresult.json', orient='records', lines=True)
        # self.df_fa = pd.read_parquet(f'{self.fld_workdata}/df_fa.parquet', engine='pyarrow')
        # self.df_fa['vector'] = self.df_fa['vector'].apply(lambda x: x.tolist()) # list列(vector)をlist型に戻す
        self.df_fa           = pd.read_json(f'{self.fld_workdata}/df_fa.json'          , orient='records', lines=True)
        self.df_cluster_info = pd.read_json(f'{self.fld_workdata}/df_cluster_info.json', orient='records', lines=True)



    # def suggest_ideas(self, target_q) :

    #--- フリーコメントに対してEmbeddingを実施（ベクトルを返す） -------------------------#
    def __myembedding_each(self, txt:str, model:str="text-embedding-3-small") :
        response = self.client.embeddings.create(input=txt, model=model)
        return response.data[0].embedding
    def __myembedding(self, sr_txt:pd.Series, model:str="text-embedding-3-small") :
        def temp_emb(txt) :
            response = self.client.embeddings.create(input=txt, model=model)
            return response.data[0].embedding
        l_vector = sr_txt.apply(temp_emb).tolist()
        return l_vector

    #--- TF-IDFでベクトル化 -----------------------------------------------------------#
    def __mytfidf(self, sr_txt:pd.Series, l_parts:list=["名詞", "動詞", "形容詞"])->pd.Series :
        # 形態素解析を行う関数を定義する
        def tokenize(text):
            mecab = MeCab.Tagger('-Ochasen')
            mecab.parse('')
            node = mecab.parseToNode(text)
            words = []
            while node:
                # 名詞、動詞、形容詞だけを抽出
                if node.feature.split(",")[0] in l_parts:
                    words.append(node.surface)
                node = node.next
            return words

        # 'text'列に対して形態素解析を行う
        tokenized_comments = sr_txt.apply(lambda x: " ".join(tokenize(x)))

        # TF-IDFでベクトル化する
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(tokenized_comments)

        # ベクトルをリスト型に変換
        l_list = X.toarray().tolist()
        return l_list


    #--- クラスター数の最適解を求める --------------------------------------------------#
    def __optimal_clustersize(self, srs:pd.Series, n_cluster_min:int, n_cluster_max:int, f_showchart:bool=False)->int :
        # クラスタリングのためのデータ準備
        X = srs.tolist()
        range_n_clusters = range(n_cluster_min, n_cluster_max)  # クラスタ数を2から9まで試す
        silhouette_scores = []

        for n_clusters in range_n_clusters:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans.fit(X)
            silhouette_scores.append(silhouette_score(X, kmeans.labels_))

        if f_showchart :
            # シルエットスコアの結果をプロット
            plt.figure(figsize=(10, 5))
            plt.plot(range_n_clusters, silhouette_scores, marker='o')
            plt.xlabel('Number of clusters')
            plt.ylabel('Silhouette Score')
            plt.title('Silhouette Analysis for Optimal Number of Clusters')
            plt.show()

        # 最適なクラスタ数を決定（例：シルエットスコアが最大のクラスタ数）
        optimal_clustersize = range_n_clusters[np.argmax(silhouette_scores)]
        return optimal_clustersize

    #--- クラスター付け ---------------------------------------------------------------#
    def __calculate_cluster(self, l_text:list, n_cluster:int) -> list :
        kmeans = KMeans(n_clusters=n_cluster, random_state=42)
        kmeans.fit(l_text)
        l_cluster = [x+1 for x in kmeans.labels_]
        return l_cluster




    #--- クラスタのタイトルを付ける(dfには、cluster/text という項目が必要) ----------------#
    def __gpt_generate_title(self, df:pd.DataFrame, model:str='gpt-4o') :

        optimal_clusters = int( df['cluster'].sort_values().to_list()[-1] )
        comments = ""
        for i in range(1, optimal_clusters+1) :
            comments += f"#クラスターNo:{i}\n"
            comments += '\n'.join('- ' + text.strip() for text in df[df['cluster'] == i]['text'])
            comments += "\n"

        prompt = f'''
            文章群に対してクラスタリングを行い分類した。それぞれのクラスターに付けるタイトルを、そこに属している文章を元に考えてほしい。
            また、そのクラスターの代表的な文章を5つ選出してください。
            JSON形式で出力してください。
            出力形式: 
            ```json
            {{
                "クラスター": [
                    {{
                        "cluster": *, 
                        "title": "" ,
                        "代表的な文章": ["", "", ・・・ ]
                    }},
                ]
            }}
            ```

            文章群: """
            {comments}
            """
        '''
        prompt = prompt.replace("            ", "").strip()

        with open(f"{self.fld_workdata}/aaaaa.txt", "w", encoding="utf-8") as f:
            f.write(prompt)

        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": ""},
                {"role": "user", "content": prompt}
            ]
        )
        message = response.choices[0].message.content
        dic = self.__myjson(message)
        with open(f'{self.fld_workdata}/clusterdata.json', 'w', encoding='utf-8') as f:
            json.dump(dic, f, ensure_ascii=False, indent=4)
        # with open('clusterdata.json', 'r', encoding='utf-8') as f:
        #     dic = json.load(f)

        df_res = pd.DataFrame( dic['クラスター'] )
        df_res = df_res.sort_values(by='cluster')
        return df_res




    #--- クラスタのタイトルを付ける(dfには、cluster/text という項目が必要) ----------------#
    def __gpt_generate_suggest(self, df:pd.DataFrame, txt_q, model:str='gpt-4o') :

        optimal_clusters = int( df['cluster'].sort_values().to_list()[-1] )
        comments = ""
        for i in range(1, optimal_clusters+1) :
            comments += f"#クラスターNo:{i}\n"
            comments += '\n'.join('- ' + text.strip() for text in df[df['cluster'] == i]['text'])
            comments += "\n"

        prompt = f'''
            あなたはコメント分析のプロです。コメントからユーザーの要望やニーズをキャッチし、最適な改善策や示唆出しをすることが得意なAIです。
            「{txt_q}」というアンケートを行い、 フリーコメントで回答を得た。
            また、それを分類分けした。
            我々が行うべき改善策などの示唆やアイデアを、それぞれの分類で３つずつ提案してください。
            示唆は箇条書きで出力してください。
            JSON形式で出力してください。
            出力形式: 
            ```json
            {{
                "示唆出し": [
                    {{
                        "cluster": *, 
                        "示唆": ["","",""]
                    }},
                ]
            }}
            ```

            フリーコメント回答: """
            {comments}
            """
        '''
        prompt = prompt.replace("            ", "").strip()

        with open(f"{self.fld_workdata}/aaaaa.txt", "w", encoding="utf-8") as f:
            f.write(prompt)

        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": ""},
                {"role": "user", "content": prompt}
            ]
        )
        message = response.choices[0].message.content
        dic = self.__myjson(message)
        with open(f'{self.fld_workdata}/clusterdata.json', 'w', encoding='utf-8') as f:
            json.dump(dic, f, ensure_ascii=False, indent=4)
        # with open('clusterdata.json', 'r', encoding='utf-8') as f:
        #     dic = json.load(f)

        df_res = pd.DataFrame( dic['示唆出し'] )
        df_res = df_res.sort_values(by='cluster')
        return df_res


    #--- JSON抽出 ------------------------------------------------------------------------------------#
    def __myjson(self, txt:str)->dict :
        pattern = r'```json(.*?)```'
        match = re.search(pattern, txt, re.DOTALL)
        dic = {}
        if match:
            json_str = match.group(1).strip()
            try:
                dic = json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f'JSONデコードエラー: {e}')
        else:
            print('マッチするパターンが見つかりませんでした。')

        return dic

    #--- 全角に変換 ---------------------------------------------------------------------#
    def __han_to_zen(self, txt:str) -> str :
        return jaconv.h2z(txt, ascii=True, digit=True)

    #--- word cloud ------------------------------------------------------------------#
    def __myworcloud(self, srs_txt:list, l_exclude_words:list, fn:str, font_path:str='C:/Windows/Fonts/meiryo.ttc', fld_save:str='static') :
        # MeCabのTaggerを初期化
        mecab = MeCab.Tagger('-u user.dic')
        # 単語カウンターの初期化
        word_counter = Counter()

        # Q2列のフリーコメントを形態素解析
        for comment in srs_txt:
            parsed = mecab.parse(comment)
            parsed_lines = parsed.split('\n')
            for line in parsed_lines:
                if line == 'EOS' or line == '':
                    continue
                word_info = line.split('\t')
                if len(word_info) > 1:
                    word = word_info[0]
                    # 品詞情報を取得
                    details = word_info[1].split(',')
                    # 名詞のみをカウント
                    if details[0] == '名詞':
                        word_counter[word] += 1

        # 除外単語をカウンターから削除
        for word in l_exclude_words:
            if word in word_counter:
                del word_counter[word]

        # 頻出単語のカウント結果を辞書に変換
        word_freq = dict(word_counter)

        # ワードクラウドの生成
        wordcloud = WordCloud(font_path=font_path, background_color='white', width=800, height=600).generate_from_frequencies(word_freq)
        if not os.path.exists(fld_save):
            os.makedirs(fld_save)
        wordcloud.to_file(f"{fld_save}/{fn}")





# #--- 固有名詞を●●（人名は▲▲）に変換する -----------------------------------------------#
# def koyuu_meishi_killer(text):
#     mecab = MeCab.Tagger('-Ochasen -u user.dic')
#     node = mecab.parseToNode(text)
#     result = []
#     while node:
#         # 品詞情報の取得
#         features = node.feature.split(',')
#         if features[0] == '名詞' and features[1] == '固有名詞':
#             if features[2] == '人名':
#                 result.append('▲▲') # 人名の場合は「▲▲」に置換
#             else:
#                 result.append('●●')
#         else:
#             # 固有名詞でない場合はそのまま
#             result.append(node.surface)
#         node = node.next
#     return ''.join(result)

