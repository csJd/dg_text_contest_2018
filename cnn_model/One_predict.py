#coding=utf-8
import CharsTool
import tensorflow as tf
import Data_helper
import numpy as np
from tensorflow.contrib import learn
import pandas as pd

#===================================================================================
#参数设置
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

#预测文件路径
tf.flags.DEFINE_string("predict_filename","../processed_data/dev_processed_data_split.csv","predict_filename path")

#vocabulary path
tf.flags.DEFINE_string("vocabulary_path","./runs/1522228533/vocab","vocabulary_path")

#model checkpoint path
tf.flags.DEFINE_string("meta_path","./runs/1522228533/checkpoints/model-1900.meta","meta_path")
tf.flags.DEFINE_string("model_path","./runs/1522228533/checkpoints/model-1900","model_path")

#result output filename
tf.flags.DEFINE_string("result_path","./result/result_predict.txt","result path")

#设置中文词语个数阈值
tf.flags.DEFINE_integer("chinese_threshold",10,"chinese_threshold")

#模型判断不出来的距离阈值
tf.flags.DEFINE_float("distance_threshold",1.0,"distance_threshold")

FLAGS = tf.flags.FLAGS

#========================================================================================

def predict(doc_words):
    '''
    :param doc_words:需要预测的文档——已经分好词的字符串
    :return:返回预测的标签 0:正常邮件 1:垃圾邮件 2：无法判别邮件
    '''
    #无法判断邮件==========================================================================
    chi_threshold = FLAGS.chinese_threshold
    is_chinese = CharsTool.is_chinese(doc_words,chi_threshold)
    is_low_num_chars = CharsTool.is_low_num_chars(doc_words,chi_threshold)

    if (not is_chinese) or (is_low_num_chars):
        return 2 #无法判别

    # 将文本映射到字典下标数组================================================================
    doc_words = [doc_words]
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(FLAGS.vocabulary_path)
    x_text = np.array(list(vocab_processor.transform(doc_words)))

    #使用CNN分类器进行二分类=================================================================
    graph = tf.Graph()

    with graph.as_default():

        session_conf = tf.ConfigProto(
            allow_soft_placement = FLAGS.allow_soft_placement,
            log_device_placement = FLAGS.log_device_placement
        )

        sess = tf.Session(config=session_conf)

        with sess.as_default():

            #模型加载
            saver = tf.train.import_meta_graph(FLAGS.meta_path)
            saver.restore(sess,FLAGS.model_path)

            #输入，输出参数
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            scores = graph.get_operation_by_name("output/scores").outputs[0]

            #预测
            pred_score = sess.run(scores,feed_dict={input_x:x_text,dropout_keep_prob:1.0})[0]

            #distance
            distance = np.abs( pred_score[0] - pred_score[1])

            #
            if distance > FLAGS.distance_threshold :

                pred_score = np.array(pred_score)

                pred = pred_score.argmax(0)

                if pred == 1:
                    pred = 0
                else:
                    pred = 1

                return pred

            else:
                return 2


if __name__ == "__main__":

    #输入
    context = "为 卖淫 猖狂 深夜 扰民 小姐 站 街 鸡头 放哨 为何 卖淫 那么 猖狂 无 相关 部门 取缔 排查 是否 其中 存在 什么 猫腻 尽管 深圳 现在 开创 文明城市 " \
              "坂田下 围东 这 一块 依然 热闹 黄色 产业 发达 站街女 穿着 暴露 眼神 挑逗 刺骨 公开 卖淫 招 嫖 厉害 已经 形成 规模 将近 年 而且 越来越 多 " \
              "已经 是 小有名气 的 红灯区 了 一条 路 至少 多个 晚上 点 半左右 开始 出来 半夜 尤其 猖狂 都 在 周围 农民房 经营 卖淫 活动 在 大 路上 " \
              "招揽生意 每次 见 男人 经过 都 会 问 玩 不 玩 整个 下围 都 乌烟瘴气 严重 影响 附近 居民 的 正常 生活 和 各 方面 的 安全隐患 有时候 想带 " \
              "朋友 亲戚 来家 做客 都 极其 不好意思 更 严重 的 是 有时 漂客 完事 不 给钱 导致 被 鸡头 殴打 事件 屡屡 发生 在 附近 经常性 听到 看到 这些 情景 难道 " \
              "又 要 到 发生 了 无可 挽救 的 事件 后 才 来 亡羊补牢 前段时间 刚 因为 卖淫 死 了 一个 人 还 不够 引起 重视 么 多次 举报 这 一块 卖淫 泛滥 问题 但 " \
              "都 不了了之 卖淫 继续 经营 泛滥 好象 是 合法 的 公开 的 存在 这是 街道 警署 不 作为 行为 还是 有 保护伞 卖淫嫖娼 既然 是 违法行为 那 为何 不 彻底 " \
              "查处 打击 都 存在 将近 年 了 有 那麼 难 吗 明眼人 都 知道 那些 站街女 卖淫嫖娼 违法 难道 街道 警署 不 知道 还是 故意 不 作为 还是 有 保护伞 强烈建议 媒体 公开 调查 报道 监督 也 强烈要求 街道 警署 真正 作为 起来 为 人民 服务 彻底 打击 下围 东 色情 卖淫 长期 泛滥 存在 的 违法行为 还 老百姓 还坂田下 围东 一片 净土 若 投诉 无果 相关 部位 仍 不 作为 本人 会 坚持 继续 一层层 上报 以及 通过 网络 曝光 坂田 地方 政府 丑陋 事实"


    print(predict(context))