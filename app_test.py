
import os
import pandas as pd
import SurveyAnalysis
import os
import time
from flask import Flask, render_template, request, session
import uuid
from flask_socketio import SocketIO, emit
from threading import Thread
from datetime import timedelta

app = Flask(__name__)
app.secret_key = 'survey_analysis_application'
app.permanent_session_lifetime = timedelta(days=7)  # セッション有効期間
socketio = SocketIO(app)

#--- インスタンス ----------------------------#
session_sas = {}

f_inprocess = False #誰かが処理しているときは、混戦しないように処理させない
fld_data = 'workdata'

@app.route('/')
def index():
    session.permanent = True  # セッションを永続化

    # index.htmlが存在しない場合にxyz.htmlをコピー
    if not os.path.exists('templates/index.html'):
        with open('templates/index_default.html', 'r', encoding='utf-8') as src_file:
            content = src_file.read()
        with open('templates/index.html', 'w', encoding='utf-8') as dest_file:
            dest_file.write(content)        

    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return render_template('index.html', session_id=session['session_id'])


@app.route('/upload', methods=['POST'])
def upload_file():
    global f_inprocess

    if f_inprocess == True :
        return "誰かが処理中です。しばらく経って再トライしてください。", 409
    
    f_inprocess = True
    file = request.files['file']
    session_id = session['session_id']
    if file:
        # ファイル名と拡張子に分けて、セッションIDを挿入
        filename = file.filename
        name, ext = os.path.splitext(filename)
        filename = f"{name}_{session_id}{ext}"
        file_path = os.path.join(fld_data, filename)
        if not os.path.exists(fld_data):
            os.makedirs(fld_data)
        file.save(file_path)

        # セッションごとのインスタンスを作成
        session_sas[session_id]    = SurveyAnalysis.SurveyAnalysis()

        # 処理を別スレッドで実行
        thread = Thread(target=long_process, args=(file_path, session_id))
        thread.start()

        # メッセージキュー確認処理も別スレッドで開始
        time.sleep(2) #SurveyAnalysisインスタンスが作られるまでちょい待ち
        queue_thread = Thread(target=process_queue, args=(session_id,))
        queue_thread.start()

        return "ファイルのアップロードが完了しました。分析処理を開始します。", 200
    return "ファイルのアップロードに失敗しました。", 409



def long_process(file_path, session_id):
    sa = session_sas[session_id]

    #--- データ読み込み ------------------------#
    df_enq = pd.read_excel( file_path, sheet_name='アンケート結果' )
    df_def = pd.read_excel( file_path, sheet_name='設定シート'    )
    df_def['f_shisa'] = False
    df_def.loc[df_def['示唆']=='〇', 'f_shisa'] = True
    df_def['f_wordcloud'] = False
    df_def.loc[df_def['ワードクラウド']=='〇', 'f_wordcloud'] = True
    sa.set_data(df_def, df_enq)

    #--- データ加工 ----------------------------------#
    sa.data_kakou()
    #--- 示唆出し ------------------------------------#
    sa.generate_suggest()
    sa.savedata()

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

    # 処理完了通知
    sa.queue_progress.put('done')

def process_queue(session_id):
    global f_inprocess
    sa = session_sas[session_id]
    
    while True:
        message = sa.queue_progress.get()
        print(message)
        if message == 'done':
            socketio.emit('done', {'sessionId': session_id, 'message': '分析が完了しました。リロードしてください。'}, namespace='/')
            break
        else:
            socketio.emit('progress', {'sessionId': session_id, 'progress': message}, namespace='/')
            time.sleep(1)

    del session_sas[session_id]

    f_inprocess = False

if __name__ == '__main__':
    socketio.run(app, debug=True)

