# coding=utf-8


# 判断是否为中文
def is_chinese(sentences, threshold):
    '''
    :param sentences: 需要判断的句子
    :param threshold: 设置的阈值，若超过threshold的中文个数值，即可认为是中文邮件
    :return:
    '''
    words_arr = sentences.strip().split()
    chiness_count = 0

    for uchar in words_arr:

        if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
            chiness_count += 1

            if chiness_count >= threshold:
                return True

    return False

# 判断邮件中文个数是否小于某个阈值
def is_low_num_chars(sentences, threshold):
    '''
    :param sentences:需要判断的句子
    :param threshold: 设置的阈值——,
    :return:
    '''
    num_count = 0
    words_arr = sentences.strip().split()
    for _ in words_arr:
        num_count += 1

        if num_count > threshold:
            return False

    return True


#Test
if __name__ == "__main__":
    # 输入
    context = "主动 深度 挖掘 全球 客户 信息 获得 优 质询 盘 您 现在 是不是 觉得 外贸 越来越 难 做 了 开发 客户 无非 就是 通过 平台 展会 平台 效果 " \
              "差 客户 质量 低询 盘 多 成交 少 利润 低 展会 门槛 高 费用 高 客户资源 单一 投入 高 但 不 一定 有 效果 想换个 开发 客户 的 方式 突围 困境 吗 " \
              "现在 网络 开发 客户 已 逐渐 成为 主流 如何 快速 出 些小 单 如何 正常 跟进 优质 客户 如何 有 节奏 的 把 询盘 变为" \
              " 有效 订单 外贸 客户 开发 系统 首席 顾问 为 您 演示 讲解 免费 课堂 详情请 加 扣扣 林先生 在线 演示 深圳 可 上门"

    context1 = "Hello world !!"

    print(is_chinese(context1, 2))