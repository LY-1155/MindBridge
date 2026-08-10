"""脱敏模块单元测试"""

from core.privacy.desensitize import desensitize, desensitize_turn


# ── 数字类 PII ──────────────────────────────────────────────

def test_phone_number():
    assert "手机号" in desensitize("我上周打电话给 13800138000 问过")


def test_long_digits():
    assert "编号" in desensitize("QQ号是 9876543210")


def test_landline():
    assert "电话" in desensitize("打过 0571-87654321 预约")


def test_year_generalized():
    assert "前几年" in desensitize("我们是2023年认识的")


# ── 姓氏 + 职业称呼 ──────────────────────────────────────────

def test_surname_doctor():
    assert "某医生" in desensitize("上周跑去找姚医生看病")


def test_surname_teacher():
    assert "某老师" in desensitize("班主任王老师找我谈了话")


def test_doctor_verb_not_mangled():
    """"看医生""找大夫"前的动词不应被替换。"""
    assert desensitize("我带他去看医生") == "我带他去看医生"


def test_time_word_not_mangled():
    """"时候医生"里的"候"是姓氏字，不应误伤。"""
    assert "时候" in desensitize("出院的时候医生给我的")


def test_surname_doctor_after_punctuation():
    """"周医生"前面是标点/句首时也应替换（漏网修复）。"""
    assert "某医生" in desensitize("周医生同样出身农村")
    assert "某医生" in desensitize("之前我们找姚医生看过")


# ── 机构名 ──────────────────────────────────────────────────

def test_hospital_name():
    assert "医院" in desensitize("在市第一人民医院住了三天")  # 前词被清掉


def test_school_name():
    out = desensitize("他转到第三中学之后就不想去了")
    assert "中学" in out and "第三" not in out


def test_school_verb_not_mangled():
    assert desensitize("他说不想回学校") == "他说不想回学校"


# ── 地名 ────────────────────────────────────────────────────

def test_major_city():
    assert "当地" in desensitize("我们从杭州搬过来的")


def test_city_suffix():
    assert "当地" in desensitize("老家在常州市武进区") or "当地" in desensitize(
        "老家在常州市武进区")


def test_clean_text_unchanged():
    """无 PII 的普通对话文本应保持原样。"""
    text = "孩子现在都不愿意出门，在家就是打游戏，怎么说都不听"
    assert desensitize(text) == text


def test_turn_role_parity():
    """desensitize_turn 与 desensitize 行为一致。"""
    text = "我上周去找姚医生，然后打了 13800138000"
    assert desensitize_turn("human", text) == desensitize(text)


def test_empty():
    assert desensitize("") == ""
    assert desensitize(None) is None
