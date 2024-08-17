
import os
import pandas as pd
import SurveyAnalysis
import os


file_path = "アンケート定義.xlsx"
sa = SurveyAnalysis.SurveyAnalysis()

#--- データ読み込み ------------------------#
df_enq = pd.read_excel( file_path, sheet_name='アンケート結果' )
df_def = pd.read_excel( file_path, sheet_name='設定シート'    )
df_def['f_shisa'] = False
df_def.loc[df_def['示唆']=='〇', 'f_shisa'] = True
df_def['f_wordcloud'] = False
df_def.loc[df_def['ワードクラウド']=='〇', 'f_wordcloud'] = True
sa.set_data(df_def, df_enq)

# #--- データ加工 ----------------------------------#
# sa.data_kakou()
# #--- 示唆出し ------------------------------------#
# sa.generate_suggest()
# sa.savedata()

sa.loaddata()
sa.set_data(df_def, df_enq)
#--- word cloud ---------------------------------#
sa.generate_wordcloud()
# --- add score ---------------------------------#
sa.add_score()
#--- generate html ------------------------------#
sa.generate_html()
#--- save data ----------------------------------#
sa.savedata()
