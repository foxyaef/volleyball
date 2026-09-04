"""V9에 네트 공중 맞대결 전용 순수 하향타를 추가 검증한 HybridRL V11 대회 봇."""

"""수비 강화학습 모델의 단일 파일 출력 템플릿.

구조
1. POLICY_WEIGHTS: 학습기가 넣는 5개 행동 × 19개 상태 특징 가중치.
2. _advance_ball/_predict: 실제 물리로 머리 높이 수신 X와 시간을 예측.
3. _features: 스냅샷과 예측값을 학습 모델 입력 19개로 변환.
4. _argmax_policy: 정지/달리기/다이빙 중 가장 높은 점수의 행동 선택.
5. decide: RIGHT를 LEFT 좌표로 대칭 변환한 뒤 결과 방향을 복원.

train.py는 가중치 자리만 숫자로 치환하므로 재학습해도 설명이 유지된다.
"""

# 학습 시 5×19 숫자 배열로 교체되는 부분이다.
POLICY_WEIGHTS = [
 [
  0.8451221974477484,
  0.22856839427916392,
  -3.619390427642594,
  0.2951778929951858,
  1.0361849550553948,
  0.31444206700007504,
  -0.6587322237666177,
  -0.3911021002863804,
  -0.5240807478440931,
  0.9156613728035854,
  0.6002451978867298,
  -0.011811506009820172,
  -1.1377014433101509,
  0.45941603146700216,
  -0.17405064832462122,
  0.07879018098738828,
  0.8301787361895017,
  0.09376484683904418,
  -0.25748298182465174
 ],
 [
  0.13395384342829436,
  -3.12133312945464,
  0.6874443796656031,
  -0.43398813680768045,
  -0.23295378471037004,
  -3.410670877253026,
  -0.3835260117497099,
  0.7059356911785637,
  0.011332042947651028,
  -0.22205047065265465,
  -0.725101839235135,
  -0.06614116815155847,
  0.6458158576438451,
  0.11206368091748145,
  1.528320226007454,
  0.19702281469977453,
  0.17987452051789715,
  0.18007351319726544,
  0.39139777694663186
 ],
 [
  -0.35542057230491114,
  3.468512105849312,
  -0.26149874953200003,
  -0.36505836070911024,
  0.026699905918958055,
  3.8484881317773003,
  -0.702063126852179,
  -0.9495664069661289,
  -1.0128405261454632,
  -0.5791062827150631,
  0.9292521214479847,
  -0.3566751562421647,
  -1.5876812993236078,
  0.37738443388837095,
  0.3003412415235323,
  0.9823774701132356,
  -0.07453499182057247,
  0.6360935468790903,
  0.12877916610018847
 ],
 [
  -2.80551467837031,
  -3.530710763048734,
  0.13315455609969,
  -1.1097167714546294,
  1.3742539869782875,
  -4.0128188714760435,
  -0.018730437000139398,
  4.582745526359511,
  0.6779508177221297,
  0.1336253540595061,
  0.7236469810933897,
  0.4101524702980017,
  -0.6123753273110369,
  1.980154901674039,
  -3.6426853749009505,
  -5.553277410204146,
  -4.00907788435609,
  -0.4510569407166135,
  -0.4776820413088792
 ],
 [
  -2.402177925013405,
  2.8302554637027417,
  -0.4432599871769754,
  -0.9595767045167939,
  0.9186513637330974,
  4.496760358206407,
  -0.5098744961953197,
  4.541135241876975,
  -0.12733272759801556,
  -0.744948213392603,
  -0.5768043769949318,
  -0.3980769638183934,
  0.4380585703531683,
  1.4607090967258898,
  -3.906796829335966,
  -5.185450457923859,
  -3.575224995795054,
  -0.1474304642925348,
  -0.13025426676105484
 ]
]

GROUND_WIDTH = 432.0
NET_X = 216.0
GROUND_Y = 252.0
PLAYER_GROUND_Y = 244.0
PLAYER_HALF = 32.0
PLAYER_MIN = 32.0
PLAYER_MAX = 184.0
PLAYER_SPEED = 6.0
BALL_MAX_Y_SPEED = 40.0
NET_HALF_WIDTH = 25.0
NET_TOP_Y = 176.0
NET_CAP_BOTTOM_Y = 192.0
HEAD_Y = 212.0

_last_canonical_direction = 0
_last_rally_frame = -1


def _clamp(value, low, high):
    """값을 물리 또는 특징의 허용 범위 안으로 제한한다."""
    return max(low, min(high, value))


def _number(value, fallback=0.0):
    """누락되거나 비정상적인 스냅샷 필드를 안전하게 숫자로 읽는다."""
    try:
        value = float(value)
        return fallback if value != value else value
    except (TypeError, ValueError):
        return fallback


def _advance_ball(x, y, vx, vy):
    """플레이어 충돌을 제외한 공 물리를 실제 엔진 순서로 1프레임 진행한다."""
    vy = _clamp(vy, -BALL_MAX_Y_SPEED, BALL_MAX_Y_SPEED)
    if x + vx < 0.0 or x + vx > GROUND_WIDTH:
        vx = -vx
    if y + vy < 0.0:
        vy = 1.0
    if abs(x - NET_X) < NET_HALF_WIDTH and y > NET_TOP_Y:
        if y <= NET_CAP_BOTTOM_Y:
            if vy > 0.0:
                vy = -vy
        elif x < NET_X:
            vx = -abs(vx)
        else:
            vx = abs(vx)
    next_y = y + vy
    if next_y > GROUND_Y:
        return x, GROUND_Y, vx, -vy, True
    return x + vx, next_y, vx, vy + 1.0, False


def _predict(x, y, vx, vy):
    """공이 서 있는 봇의 머리 높이에 도착할 (X, 남은 프레임)을 찾는다."""
    for eta in range(1, 241):
        old_y = y
        x, y, vx, vy, ground = _advance_ball(x, y, vx, vy)
        if y > old_y and y >= HEAD_Y and 0.0 <= x <= NET_X:
            return _clamp(x, PLAYER_MIN, PLAYER_MAX), eta
        if ground:
            return None
    return None


def _features(self_x, state, ball_x, ball_y, ball_vx, ball_vy):
    """현재 물리 상태를 학습 정책이 사용하는 정규화 특징 19개로 바꾼다."""
    prediction = _predict(ball_x, ball_y, ball_vx, ball_vy)
    if prediction is None:
        target_x, eta = self_x, 30
    else:
        target_x, eta = prediction
    dx = target_x - self_x
    useful_frames = max(1.0, eta - 1.0)
    required = dx / (PLAYER_SPEED * useful_frames)
    normal_gap = abs(dx) - PLAYER_SPEED * useful_frames - PLAYER_HALF
    last = float(_last_canonical_direction)
    toward = 1.0 if dx * last > 0.0 else (-1.0 if dx * last < 0.0 else 0.0)
    return [
        1.0,
        _clamp(dx / 96.0, -2.0, 2.0),
        _clamp(abs(dx) / 96.0, 0.0, 2.0),
        _clamp(eta / 24.0, 0.0, 2.0),
        1.0 / max(1.0, eta / 4.0),
        _clamp(required, -2.0, 2.0),
        _clamp(abs(required), 0.0, 2.0),
        _clamp(normal_gap / 48.0, -2.0, 2.0),
        last,
        toward,
        (self_x - 108.0) / 76.0,
        _clamp(ball_vx / 40.0, -2.0, 2.0),
        _clamp(ball_vy / 40.0, -1.0, 1.0),
        1.0 if state == 0 else 0.0,
        1.0 if state in (1, 2) else 0.0,
        1.0 if state == 3 else 0.0,
        1.0 if state == 4 else 0.0,
        _clamp((56.0 - self_x) / 24.0, 0.0, 1.0),
        _clamp((self_x - 160.0) / 24.0, 0.0, 1.0),
    ]


def _argmax_policy(features, state):
    """가중합 점수가 최대인 행동을 고르고 현재 불가능한 다이빙은 제외한다."""
    best_action = 0
    best_score = -1.0e100
    for action, row in enumerate(POLICY_WEIGHTS):
        if state != 0 and action >= 3:
            continue
        score = 0.0
        for weight, feature in zip(row, features):
            score += weight * feature
        if score > best_score:
            best_score = score
            best_action = action
    return best_action


def decide(s):
    """수비 진입점: 점프/공격 없이 학습된 정지·달리기·다이빙을 반환한다."""
    global _last_canonical_direction, _last_rally_frame

    side = s.get('side', 'LEFT')
    me = s.get('self', {})
    ball = s.get('ball', {})
    meta = s.get('meta', {})
    rally = int(_number(meta.get('rallyFrameCount'), 0))
    if rally < _last_rally_frame or rally <= 1:
        _last_canonical_direction = 0
    _last_rally_frame = rally

    # Mirror RIGHT into the LEFT coordinate system used during training.
    mirror = -1 if side == 'RIGHT' else 1
    self_x_raw = _number(me.get('x'), 108 if side == 'LEFT' else 324)
    ball_x_raw = _number(ball.get('x'))
    self_x = GROUND_WIDTH - self_x_raw if mirror == -1 else self_x_raw
    ball_x = GROUND_WIDTH - ball_x_raw if mirror == -1 else ball_x_raw
    ball_vx = _number(ball.get('xVelocity')) * mirror
    ball_y = _number(ball.get('y'))
    ball_vy = _number(ball.get('yVelocity'))
    state = int(_number(me.get('state'), 0))

    landing_raw = _number(ball.get('expectedLandingPointX'), -9999.0)
    landing = GROUND_WIDTH - landing_raw if mirror == -1 else landing_raw
    incoming = 0.0 <= landing <= NET_X

    if incoming:
        action = _argmax_policy(
            _features(self_x, state, ball_x, ball_y, ball_vx, ball_vy), state
        )
    else:
        # If the opponent is preparing a near-net hit, wait a little forward
        # of centre.  The V9 defense was trained from this ready position on an
        # even mix of short spikes and deep/back shots.  Otherwise keep V7's
        # neutral centre so ordinary defense is unchanged.
        opp = s.get('opp', {})
        opp_x_raw = _number(opp.get('x'), 324 if side == 'LEFT' else 108)
        opp_x = GROUND_WIDTH - opp_x_raw if mirror == -1 else opp_x_raw
        opponent_near_net = 248.0 <= opp_x <= 286.0
        ball_on_opponent_side = ball_x > NET_X
        ready_x = 118.0 if opponent_near_net and ball_on_opponent_side else 108.0
        dx = ready_x - self_x
        action = 2 if dx > 6.0 else (1 if dx < -6.0 else 0)

    if action == 1:
        canonical_x, hit = -1, 0
    elif action == 2:
        canonical_x, hit = 1, 0
    elif action == 3:
        canonical_x, hit = -1, 1
    elif action == 4:
        canonical_x, hit = 1, 1
    else:
        canonical_x, hit = 0, 0

    if state != 0:
        hit = 0
    _last_canonical_direction = canonical_x
    return {'x': int(canonical_x * mirror), 'y': 0, 'hit': int(hit)}


# 기존 decide를 고정 수비 모듈로 보존한다.
_DEFENSE_DECIDE = decide

"""학습된 기본 공격과 당일 스킬 후보를 같은 형식으로 다루는 단일 모듈.

이 파일은 아직 decide() 전체가 아니다. 통합 봇에서 수비가 머리 위 제어를
완료했다고 판단했을 때 ``attack_decision(snapshot)``을 호출한다.
"""

ATTACK_WEIGHTS = [
 0.875195681216235,
 -0.9699921043721251,
 0.4638505539428905,
 -0.20929259221809265,
 -0.6056474314952905,
 1.0615303106562828,
 -1.0383841584651583,
 1.6201949559140727,
 -0.4555785611549872,
 0.1580655757152867,
 -1.5276195049092791,
 3.261774095931002,
 -3.1432493106315365,
 -0.007970886603498727,
 -0.19069788065329663,
 0.10247309226337699,
 -7.993759026400874,
 0.0008690682427065916,
 0.08936954533025776,
 -0.12359956154185503,
 0.0065496836471188175,
 -0.20888424041070203,
 -0.2733013642177412,
 -0.08648347782017914,
 0.051163721110456216,
 -0.03337350616747137,
 -8.091010231955611,
 0.15197572811853777,
 -0.05278758793905838,
 0.06320860456251795,
 -0.23119591640382622,
 -0.00031937598364642494,
 0.042427600568066604,
 0.2661802865248382,
 0.05936100610771843,
 -0.10616122812894674,
 1.1414406440691844,
 1.370044003924856,
 -0.2755873298837206,
 0.1296297109213634,
 -1.7136828782193954,
 0.06647212435171068,
 -0.07359747642407612,
 0.093532023905357,
 -0.08708309168984114,
 0.097443226915872,
 0.024904874800457062,
 2.022539796531337,
 -0.22343485942229507,
 0.44725523059620775,
 0.07438226219729777,
 -0.3907607503662719,
 -0.055486507326844035,
 0.0543896222921198,
 -0.0016105985345847163,
 0.270074057607447,
 -0.5844293562019556,
 1.9124976295198421,
 -0.551910198094795,
 0.3250708277974991,
 1.4146771560387954,
 0.048543875651937435,
 0.005748864412442819,
 -0.11575270556923664,
 -0.05223817079282628,
 0.0374032596845649
]

SHOT_TYPES = (
    (0, -1, 'slow_up'),
    (0, 0, 'slow_flat'),
    (0, 1, 'slow_down'),
    (1, -1, 'fast_up'),
    (1, 0, 'fast_flat'),
    (1, 1, 'fast_down'),
)
SHOT_FEATURE_COUNT = 10
SETTING_COUNT = 6


def _settings():
    """학습된 앞 6개 값을 정렬·점프·타격 범위로 변환한다."""
    p = ATTACK_WEIGHTS
    return {
        'lead': _clamp(4.0 + p[0] * 2.0, 0.0, 10.0),
        'deadband': _clamp(6.0 + p[1] * 3.0, 1.0, 20.0),
        'jump_vy': _clamp(-1.0 + p[2] * 3.0, -10.0, 8.0),
        'jump_y': _clamp(140.0 + p[3] * 24.0, 55.0, 210.0),
        'arm_x': _clamp(38.0 + p[4] * 8.0, 16.0, 70.0),
        'arm_y': _clamp(72.0 + p[5] * 14.0, 30.0, 130.0),
    }


def _project_shot(ball_x, ball_y, ball_vy, x_input, y_input):
    """느린/빠른 × 위/수평/아래 후보의 착지점과 비행시간을 예측한다."""
    x, y = ball_x, ball_y
    vx = (abs(x_input) + 1) * 10.0
    vy = abs(ball_vy) * y_input * 2.0
    crossed = False
    for frame in range(1, 181):
        x, y, vx, vy, ground = _advance_ball(x, y, vx, vy)
        if x > NET_X:
            crossed = True
        if ground:
            return crossed and x > NET_X, x, frame
    return False, x, 180


def _shot_features(landing_x, flight, self_x, opponent_x, opponent_state):
    """후보 착지와 현재 상대 위치·상태를 10개 공격 특징으로 만든다."""
    opponent_front = _clamp((324.0 - opponent_x) / 76.0, -1.0, 1.0)
    vulnerable = 1.0 if opponent_state in (2, 3, 4) else 0.0
    return [
        1.0,
        abs(landing_x - opponent_x) / 152.0,
        flight / 60.0,
        (landing_x - 324.0) / 108.0,
        (self_x - 108.0) / 76.0,
        (opponent_x - 324.0) / 76.0,
        opponent_front,
        vulnerable,
        (landing_x - opponent_x) / 152.0,
        1.0 if opponent_state == 0 else -1.0,
    ]


def _choose_shot(ball_x, ball_y, ball_vy, self_x, opponent_x, opponent_state):
    """현재 상대 위치에 맞춰 여섯 공격 후보 중 학습 점수가 가장 큰 것을 선택."""
    best = (1, -1, 'fast_up')
    best_score = -1.0e100
    for index, shot in enumerate(SHOT_TYPES):
        x_input, y_input, _ = shot
        crosses, landing_x, flight = _project_shot(
            ball_x, ball_y, ball_vy, x_input, y_input
        )
        if not crosses:
            continue
        features = _shot_features(
            landing_x, flight, self_x, opponent_x, opponent_state
        )
        start = SETTING_COUNT + index * SHOT_FEATURE_COUNT
        row = ATTACK_WEIGHTS[start:start + SHOT_FEATURE_COUNT]
        score = sum(weight * feature for weight, feature in zip(row, features))
        if score > best_score:
            best, best_score = shot, score
    return best[0], best[1], best[2], best_score


def _attack_future_x(ball_x, ball_vx, lead):
    """벽 반사가 임박한 경우에만 학습된 v5 보정으로 공격 정렬 X를 계산한다.

    일반 상황은 v4의 직선 예측을 한 글자도 다르게 해석하지 않는다. 직선
    예측이 뒤쪽 벽(x=0)을 통과할 때에만 실제 엔진 순서로 반사를 적용하고,
    벽 전용 표본에서 고른 선행시간 1.6배와 코트 안쪽 6px 보정을 사용한다.
    """
    straight_x = ball_x + ball_vx * lead
    # 기본 AI 상대 실점 로그에서 첫 수비 뒤 vx=-1인 공에 v5의 고속
    # 반사 보정이 켜져 두 번째 접촉을 놓치는 사례가 집중됐다. 저속 공은
    # 벽 x=32를 지키고, 속도 2 이상일 때만 v5 반사 추적을 사용한다.
    if ball_vx > -2.0 or straight_x >= 0.0:
        return _clamp(straight_x, PLAYER_MIN, PLAYER_MAX)

    wall_lead = lead * 1.6
    whole = int(max(0.0, wall_lead))
    fraction = max(0.0, wall_lead) - whole
    x, vx = ball_x, ball_vx
    for _ in range(whole):
        if x + vx < 0.0 or x + vx > GROUND_WIDTH:
            vx = -vx
        x += vx
    if fraction:
        if x + vx * fraction < 0.0 or x + vx * fraction > GROUND_WIDTH:
            vx = -vx
        x += vx * fraction
    return _clamp(x + 6.0, PLAYER_MIN, PLAYER_MAX)


def basic_attack_candidate(s):
    """수비 완료 후 위치별 6패턴 공격을 공통 후보 형식으로 반환한다."""
    side = s.get('side', 'LEFT')
    mirror = -1 if side == 'RIGHT' else 1
    me = s.get('self', {})
    opp = s.get('opp', {})
    ball = s.get('ball', {})
    self_x_raw = _number(me.get('x'), 108 if mirror == 1 else 324)
    opp_x_raw = _number(opp.get('x'), 324 if mirror == 1 else 108)
    ball_x_raw = _number(ball.get('x'))
    self_x = GROUND_WIDTH - self_x_raw if mirror == -1 else self_x_raw
    opponent_x = GROUND_WIDTH - opp_x_raw if mirror == -1 else opp_x_raw
    ball_x = GROUND_WIDTH - ball_x_raw if mirror == -1 else ball_x_raw
    ball_vx = _number(ball.get('xVelocity')) * mirror
    ball_y = _number(ball.get('y'))
    ball_vy = _number(ball.get('yVelocity'))
    self_y = _number(me.get('y'), 244)
    opponent_y = _number(opp.get('y'), 244)
    state = int(_number(me.get('state'), 0))
    opponent_state = int(_number(opp.get('state'), 0))
    settings = _settings()

    future_x = _attack_future_x(ball_x, ball_vx, settings['lead'])
    dx = future_x - self_x
    align = 1 if dx > settings['deadband'] else (-1 if dx < -settings['deadband'] else 0)

    if state in (3, 4):
        canonical_action = {'x': align, 'y': 0, 'hit': 0}
        pattern_name, score = 'recover', 0.2
    elif state == 0:
        jump_now = ball_y <= settings['jump_y'] and ball_vy >= settings['jump_vy']
        canonical_action = {'x': align, 'y': -1 if jump_now else 0, 'hit': 0}
        pattern_name, score = 'jump_setup', 0.55
    else:
        close = (
            abs(ball_x - self_x) <= settings['arm_x']
            and abs(ball_y - self_y) <= settings['arm_y']
        )
        # Learned V11 specialist: both players are airborne, almost level, and
        # touching the net-side attack point.  V9's fast horizontal shot gives
        # the opponent an immediate downward counter here.  Pure input +Y
        # deliberately uses x=0, y=+1 only inside this narrow held-out-tested
        # region; every other state still uses the complete V9 attack policy.
        net_clash_down = (
            close
            and state in (1, 2)
            and opponent_state in (1, 2)
            and self_x >= 178.0
            and opponent_x <= 286.0
            and opponent_x - self_x <= 92.0
            and abs(self_y - opponent_y) <= 8.0
            and abs(ball_y - self_y) <= 12.0
        )
        if net_clash_down:
            shot_x, shot_y = 0, 1
            pattern_name, shot_score = 'net_clash_down', 2.0
        else:
            shot_x, shot_y, pattern_name, shot_score = _choose_shot(
                ball_x, ball_y, ball_vy, self_x, opponent_x, opponent_state
            )
        strike_x = shot_x if close else align
        # x 입력의 절댓값이 파워샷 속도를 정하므로, 뒤쪽 벽에서는
        # 빠른 슛(+1)을 같은 세기의 -1로 바꿔 벽에 몸을 고정한다.
        # 공과 겹치기 직전에 안쪽으로 6px 이동해 접촉을 놓치는 것을 막는다.
        if (
            close and ball_x < PLAYER_MIN
            and self_x <= PLAYER_MIN + PLAYER_SPEED
            and abs(shot_x) == 1
        ):
            strike_x = -1
        canonical_action = {
            'x': strike_x,
            'y': int(shot_y),
            'hit': 1,
        }
        score = 0.7 + min(max(shot_score, -2.0), 2.0) * 0.05

    action = {
        'x': int(canonical_action['x'] * mirror),
        'y': int(canonical_action['y']),
        'hit': int(canonical_action['hit']),
    }
    return {
        'name': 'diverse_attack_v11_' + pattern_name,
        'available': True,
        'feasible': True,
        'score': score,
        'action': action,
    }



def skill_candidates(s):
    """대회 당일 새 스킬 후보를 추가하는 유일한 함수.

    새 필드는 반드시 ``s.get('필드명', 기본값)``으로 읽는다. 각 스킬도
    basic_attack_candidate와 같은 name/available/feasible/score/action 구조로
    append하면 된다. 스킬이 없거나 코드를 지우면 기본 공격은 그대로 작동한다.
    """
    candidates = []
    # 예시:
    # can_dash = bool(s.get('self', {}).get('canDash', False))
    # candidates.append({
    #     'name': 'dash_attack', 'available': can_dash,
    #     'feasible': can_dash and ..., 'score': 0.9,
    #     'action': {'x': 1, 'y': 0, 'hit': 1, 'dash': 1},
    # })
    return candidates


def attack_decision(s):
    """기본 공격과 당일 스킬 중 사용 가능하고 점수가 가장 높은 행동 반환."""
    candidates = [basic_attack_candidate(s)] + skill_candidates(s)
    usable = [
        item for item in candidates
        if item.get('available', False) and item.get('feasible', False)
    ]
    if not usable:
        return {'x': 0, 'y': 0, 'hit': 0}
    return max(usable, key=lambda item: item.get('score', 0.0))['action']


# ===========================================================================
# 수비-공격 전환 관리자
# ===========================================================================
# 위에서 보존한 _DEFENSE_DECIDE와 attack_decision 중 하나만 호출한다.
# 스냅샷에는 명시적인 "내가 공을 만졌다" 필드가 없으므로, 연속 스냅샷에서
# 공이 하강(+Y)하다 상승(-Y)으로 바뀌고 우리 캐릭터 가까이에 있었는지 본다.
_hybrid_previous = None
_hybrid_touches = 0
_hybrid_attack_mode = False
_hybrid_last_rally = -1


def _hybrid_observation(s):
    me = s.get('self', {})
    ball = s.get('ball', {})
    return {
        'self_x': _number(me.get('x')),
        'self_y': _number(me.get('y'), 244),
        'ball_x': _number(ball.get('x')),
        'ball_y': _number(ball.get('y')),
        'ball_vx': _number(ball.get('xVelocity')),
        'ball_vy': _number(ball.get('yVelocity')),
    }


def _hybrid_our_touch(s):
    """네트 반사가 아니라 우리 캐릭터의 일반 수비 충돌인지 판정한다."""
    if _hybrid_previous is None:
        return False
    old = _hybrid_previous
    now = _hybrid_observation(s)
    ball = s.get('ball', {})
    return (
        old['ball_vy'] > 0
        and now['ball_vy'] < 0
        and min(
            abs(old['ball_x'] - old['self_x']),
            abs(now['ball_x'] - now['self_x']),
        ) <= 44
        and min(
            abs(old['ball_y'] - old['self_y']),
            abs(now['ball_y'] - now['self_y']),
        ) <= 56
        and not bool(ball.get('isPowerHit', False))
    )


def _hybrid_opponent_return(s):
    """상대 코트에서 우리 쪽으로 향하는 새 공격이면 터치 카운트를 초기화."""
    side = s.get('side', 'LEFT')
    ball = s.get('ball', {})
    x = _number(ball.get('x'))
    vx = _number(ball.get('xVelocity'))
    if side == 'RIGHT':
        return x < 216 and vx > 0
    return x > 216 and vx < 0


def decide(s):
    """최종 진입점: 수비 제어 전에는 DefenseRL, 제어 후에는 AttackRL."""
    global _hybrid_previous, _hybrid_touches, _hybrid_attack_mode
    global _hybrid_last_rally, _last_canonical_direction

    meta = s.get('meta', {})
    rally = int(_number(meta.get('rallyFrameCount'), 0))
    new_rally = rally < _hybrid_last_rally or rally <= 1
    if new_rally or _hybrid_opponent_return(s):
        _hybrid_previous = None if new_rally else _hybrid_previous
        _hybrid_touches = 0
        _hybrid_attack_mode = False

    if _hybrid_our_touch(s):
        _hybrid_touches += 1
        ball_vx = abs(_number(s.get('ball', {}).get('xVelocity')))
        # 첫 터치라도 이미 수직 제어됐으면 바로 공격한다. 그렇지 않으면
        # DefenseRL이 한 번 더 받고, 두 번째 터치 후 공격으로 넘어간다.
        if ball_vx <= 1.0 or _hybrid_touches >= 2:
            _hybrid_attack_mode = True

    if bool(s.get('ball', {}).get('isPowerHit', False)):
        _hybrid_touches = 0
        _hybrid_attack_mode = False

    if _hybrid_attack_mode:
        action = attack_decision(s)
        # 공격 중 반환한 X도 다음 수비의 Worker 지연 특징에 이어 준다.
        mirror = -1 if s.get('side', 'LEFT') == 'RIGHT' else 1
        _last_canonical_direction = int(_number(action.get('x'))) * mirror
    else:
        action = _DEFENSE_DECIDE(s)

    _hybrid_previous = _hybrid_observation(s)
    _hybrid_last_rally = rally
    return action
