import random
import re
from transformers import pipeline

# 1. 설정값
MODEL_NAME = "klue/bert-base"  # 한국어 BERT 모델 (빈칸 맞히기 학습됨)
TOP_K = 5                       # AI 예측 후보를 몇 개까지 보여줄지
ROUNDS = 5                      # 맞히기 모드 문제 수
TRICK_ROUNDS = 3                # 속이기 모드 문제 수
TRICK_POINTS = {2: 3, 3: 2, 4: 1, 5: 1}  # 속이기 모드: 정답 순위별 점수

# 2. 모델 불러오기 (처음 한 번은 다운로드 때문에 시간이 걸림)
print("AI 모델을 불러오는 중입니다... (처음 실행할 때는 몇 분 걸릴 수 있어요)")
fill = pipeline("fill-mask", model=MODEL_NAME, top_k=TOP_K)
tokenizer = fill.tokenizer
MASK = tokenizer.mask_token  # 이 모델이 빈칸을 나타내는 기호 (klue는 [MASK])

# 3. 맞히기 모드 문제 파일 (한 줄에 한 문제, 정답을 [ ]로 감싸서 작성)
PROBLEM_FILE = "problems.txt"


# ---------------- 공통 함수 ----------------

def is_one_token(word):
    """모델이 빈칸 하나로 맞힐 수 있는 단어인지 확인 (토큰 1개짜리 단어만 가능)"""
    tokens = tokenizer.tokenize(word)
    return len(tokens) == 1 and tokens[0] != tokenizer.unk_token


def to_model_text(sentence):
    """[MASK] 표시를 모델이 쓰는 빈칸 기호로 바꿈"""
    return sentence.replace("[MASK]", MASK)


def ai_predict(sentence):
    """AI가 예측한 빈칸 후보와 확률 목록을 돌려줌"""
    results = fill(to_model_text(sentence))
    return [(r["token_str"].replace("##", "").strip(), r["score"]) for r in results]


def ai_score_of(sentence, word):
    """AI가 특정 단어를 몇 % 확률로 예상했는지 돌려줌 (계산할 수 없으면 None)"""
    if not word or not is_one_token(word):
        return None
    result = fill(to_model_text(sentence), targets=[word])
    return result[0]["score"]


def parse_problem(text):
    """'고래 싸움에 [새우] 등 터진다.' → ('고래 싸움에 [MASK] 등 터진다.', '새우', None)
    문제가 있으면 (None, None, 이유)를 돌려줌"""
    found = re.findall(r"\[(.+?)\]", text)  # [ ] 안의 글자 찾기
    if len(found) != 1 or found[0].strip().upper() == "MASK":
        return None, None, "정답 단어 하나를 [ ]로 감싸 주세요"

    answer = found[0].strip()
    if not is_one_token(answer):
        pieces = tokenizer.tokenize(answer)
        return None, None, f"정답 '{answer}'은(는) {pieces}처럼 여러 조각이라 AI가 맞힐 수 없어요"

    sentence = text.replace(f"[{found[0]}]", "[MASK]", 1)  # 정답 자리를 빈칸으로
    return sentence, answer, None


def load_problems(path):
    """문제 파일을 읽어서 [{"q": 문장, "a": 정답}, ...] 목록으로 만듦"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"[알림] 문제 파일을 찾을 수 없습니다: {path}")
        return []

    problems = []
    skipped = []
    for line_no, line in enumerate(lines, start=1):
        text = line.strip()
        if not text or text.startswith("#"):  # 빈 줄과 주석은 건너뜀
            continue

        sentence, answer, error = parse_problem(text)
        if error:
            skipped.append(f"  {line_no}번째 줄: {text}  → {error}")
        else:
            problems.append({"q": sentence, "a": answer})

    if skipped:
        print(f"[알림] 문제 {len(skipped)}개는 사용할 수 없어서 제외했습니다.")
        for msg in skipped:
            print(msg)

    return problems


def find_rank(predictions, answer):
    """AI 후보 목록에서 정답이 몇 위인지 찾음 (없으면 None)"""
    for rank, (word, score) in enumerate(predictions, start=1):
        if word == answer:
            return rank
    return None


def bar(score, width=20):
    """확률을 ████ 막대로 표시"""
    return "█" * max(1, round(score * width))


def show_predictions(predictions, answer, label):
    """AI 후보 목록을 순위, 막대, 확률과 함께 출력"""
    print("\nAI가 생각한 후보:")
    for rank, (word, score) in enumerate(predictions, start=1):
        mark = f" ← {label}" if word == answer else ""
        print(f"  {rank}위 {word:<8} {bar(score)} {score * 100:.1f}%{mark}")


# ---------------- 맞히기 모드 ----------------

def quiz_mode():
    """AI가 낸 빈칸 문제를 나와 AI가 함께 푸는 모드"""
    # 문제 파일 불러오기 (매번 새로 읽어서, 게임 중에 파일을 고쳐도 바로 반영됨)
    problems = load_problems(PROBLEM_FILE)
    if not problems:
        print(f"풀 수 있는 문제가 없습니다. {PROBLEM_FILE} 파일을 확인해 주세요.")
        return

    random.shuffle(problems)
    rounds = min(ROUNDS, len(problems))

    my_score = 0
    ai_score = 0

    print("\n=== 🧠 맞히기 모드: AI vs 나 ===")
    print(f"총 {rounds}문제! 빈칸(____)에 들어갈 단어를 맞혀 보세요.")
    print("모르면 그냥 엔터, 그만하려면 exit\n")

    for i, p in enumerate(problems[:rounds], start=1):
        sentence = p["q"]
        answer = p["a"]

        print(f"[{i}/{rounds}] {sentence.replace('[MASK]', '____')}")
        my_answer = input("나의 답 : ").strip()

        if my_answer == "exit":
            print("맞히기 모드를 중단합니다.")
            break

        predictions = ai_predict(sentence)
        ai_answer = predictions[0][0]

        # 채점
        i_correct = my_answer == answer
        ai_correct = ai_answer == answer
        if i_correct:
            my_score += 1
        if ai_correct:
            ai_score += 1

        print(f"\n정답 : {answer}")
        print(f"나   : {my_answer or '(패스)'} {'⭕' if i_correct else '❌'}")
        print(f"AI   : {ai_answer} {'⭕' if ai_correct else '❌'}")

        show_predictions(predictions, answer, "정답")

        my_word_score = ai_score_of(sentence, my_answer)
        if my_word_score is not None:
            print(f"\n💡 AI는 내 답 '{my_answer}'을(를) {my_word_score * 100:.2f}% 확률로 예상했어요.")

        print(f"\n현재 점수 → 나 {my_score} : {ai_score} AI")
        print("-" * 40 + "\n")

    # 최종 결과
    print(f"=== 맞히기 모드 결과: 나 {my_score} : {ai_score} AI ===")
    if my_score > ai_score:
        print("🎉 AI를 이겼습니다!")
    elif my_score < ai_score:
        print("🤖 AI의 승리! 다음엔 이겨 봐요.")
    else:
        print("🤝 무승부!")


# ---------------- 속이기 모드 ----------------

def ask_problem():
    """'핸드폰을 많이 하면 [눈]이 아프다.'처럼 정답을 [ ]로 감싼 문제를 받음
    (문장, 정답)을 돌려주고, exit면 (None, None)"""
    while True:
        text = input("문제 : ").strip()
        if text == "exit":
            return None, None

        sentence, answer, error = parse_problem(text)
        if error:
            print(f"  ⚠️ {error}. 예: 핸드폰을 많이 하면 [눈]이 아프다.")
            continue
        return sentence, answer


def trick_mode():
    """내가 문제를 내고 AI가 푸는 모드"""
    total = 0

    print("\n=== 😈 속이기 모드: 내가 문제 내기 ===")
    print("사람은 쉽게 맞히지만 AI는 헷갈리는 문장을 만들어 보세요!")
    print("내 정답이 AI 후보 2~5위면 성공, 1위면 실패, 5위 밖이면 무효예요.")
    print("점수: 2위 3점 / 3위 2점 / 4~5위 1점")
    print("정답 단어를 [ ]로 감싸서 한 줄로 입력하세요. 예: 핸드폰을 많이 하면 [눈]이 아프다.")
    print("그만하려면 exit\n")

    for i in range(1, TRICK_ROUNDS + 1):
        print(f"[{i}/{TRICK_ROUNDS}]")
        sentence, answer = ask_problem()
        if sentence is None:
            print("속이기 모드를 중단합니다.")
            break
        print(f"AI에게 낸 문제: {sentence.replace('[MASK]', '____')}")

        predictions = ai_predict(sentence)
        rank = find_rank(predictions, answer)

        show_predictions(predictions, answer, "내 정답")

        if rank == 1:
            print("\n❌ 실패! AI가 정답을 맞혔어요.")
        elif rank is None:
            my_score = ai_score_of(sentence, answer)
            print(f"\n⚠️ 무효! AI는 '{answer}'을(를) {my_score * 100:.3f}%로밖에 예상하지 못했어요.")
            print("   너무 엉뚱한 문장이에요. 사람이 봐도 맞힐 수 있게 만들어 보세요.")
        else:
            points = TRICK_POINTS[rank]
            total += points
            print(f"\n🎉 성공! 내 정답이 AI 후보 {rank}위였어요. +{points}점")

        print(f"현재 점수: {total}점")
        print("-" * 40 + "\n")

    print(f"=== 속이기 모드 결과: {total}점 ===")


# ---------------- 메인 메뉴 ----------------

while True:
    print("\n=== 🧠 빈칸 퀴즈 ===")
    print("1. 맞히기 모드 (AI vs 나)")
    print("2. 속이기 모드 (내가 문제 내기)")
    print("3. 종료")
    choice = input("선택 : ").strip()

    if choice == "1":
        quiz_mode()
    elif choice == "2":
        trick_mode()
    elif choice == "3" or choice == "exit":
        print("게임을 종료합니다.")
        break
    else:
        print("1, 2, 3 중에서 골라 주세요.")
