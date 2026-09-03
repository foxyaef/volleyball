"""v1에서 추가 학습하고 독립 검증한 DefenseRL + AttackRL v2 대회 봇."""

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
POLICY_WEIGHTS = [[0.993817894222016,
  0.3671145133544119,
  -3.507579265272672,
  0.15586688352715827,
  1.1858873688944942,
  0.2058245263766206,
  -0.6793405267610154,
  -0.27030918919192004,
  -0.563806756539287,
  0.8910099523179391,
  0.5339132191337392,
  0.08912852102428338,
  -1.2135729953626575,
  0.43454478316593514,
  -0.10205527083979568,
  0.03235069229763193,
  1.0182452648524196,
  -0.07333097087082296,
  -0.41399943376173615],
 [0.12841324360902173,
  -3.158298319727894,
  0.45094355113831214,
  -0.37716043637886665,
  -0.2856029958093599,
  -3.3748773073927896,
  -0.3238229963044144,
  0.5628556544998855,
  0.11937526419462369,
  -0.32846711156548297,
  -0.6113244198586016,
  -0.0946227132129008,
  0.6149661773785872,
  0.08527887298246033,
  1.471422615956202,
  0.07788408910629485,
  0.18328812267178118,
  0.3200928903391416,
  0.3539483861485896],
 [-0.23561823462327675,
  3.5223435286883955,
  -0.30488888481936444,
  -0.5272714301633331,
  0.09844679254690875,
  3.867163364969045,
  -0.5238487652878426,
  -0.814386237635357,
  -1.0140296560606132,
  -0.6248933627659287,
  1.0291693505296182,
  -0.29924241431169685,
  -1.4241054433004043,
  0.40238794351743934,
  0.20553637182146953,
  0.9132343440603473,
  -0.11940966420721633,
  0.6020876684850816,
  -0.02003992408497331],
 [-2.9228650549454684,
  -3.5605079882504262,
  0.022388791822404136,
  -1.106332463531204,
  1.3983020005284565,
  -4.111157681795052,
  -0.18062080526254293,
  4.27362933914553,
  0.6398596410049102,
  -0.029854174408527227,
  0.9497860775450396,
  0.7044493499129106,
  -0.5544589384751492,
  1.9828358589661967,
  -3.7999455680835683,
  -5.6672312803219596,
  -3.927632883278716,
  -0.43946390328538354,
  -0.36025254467317913],
 [-2.428263231289848,
  2.5464686055941974,
  -0.22392004673892024,
  -1.0492569278374813,
  0.8668978203504722,
  4.348638933555156,
  -0.45112472223895916,
  4.776623842921827,
  -0.2506064787530416,
  -0.5354755742467859,
  -0.7961954585246551,
  -0.35220114016883236,
  0.34186418115781325,
  1.611819863205277,
  -3.968676033927934,
  -5.281474990911492,
  -3.8296797303445107,
  -0.039854893538002076,
  -0.5853809802873637]]

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
        # No threat: recover to the canonical centre so the next randomized
        # attack starts from the most useful position.
        dx = 108.0 - self_x
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

ATTACK_WEIGHTS = [1.1115192835528236,
 -0.9740376842501999,
 0.4628828059785667,
 -0.04936270154689246,
 -1.2657298711197869,
 0.9559610756091674,
 1.2615515523890066,
 1.507380380967121,
 -0.1484075836609828,
 0.22730985703104134,
 -1.800009626537353,
 0.0705659030588246,
 2.025719694248445,
 -0.1859063551969145,
 0.40815963546004835,
 0.4380447607873059,
 -0.8908122710866784,
 1.6811144773019557,
 -0.5732539415122792,
 0.38188307490312384,
 2.2594402078064078]

GROUND_WIDTH = 432.0
NET_X = 216.0
GROUND_Y = 252.0
BALL_MAX_Y_SPEED = 40.0
NET_HALF_WIDTH = 25.0
NET_TOP_Y = 176.0
NET_CAP_BOTTOM_Y = 192.0
SHOT_TYPES = (-1, 0, 1)  # 위, 수평, 아래 파워샷


def _clamp(value, low, high):
    return max(low, min(high, value))


def _number(value, fallback=0.0):
    """기존/추가 스냅샷 필드를 안전하게 읽기 위한 공통 변환 함수."""
    try:
        value = float(value)
        return fallback if value != value else value
    except (TypeError, ValueError):
        return fallback


def _settings():
    """학습 가중치 앞 6개를 사람이 이해할 수 있는 공격 타이밍 값으로 변환."""
    p = ATTACK_WEIGHTS
    return {
        'lead': _clamp(4.0 + p[0] * 2.0, 0.0, 10.0),
        'deadband': _clamp(6.0 + p[1] * 3.0, 1.0, 20.0),
        'jump_vy': _clamp(-1.0 + p[2] * 3.0, -10.0, 8.0),
        'jump_y': _clamp(140.0 + p[3] * 24.0, 55.0, 210.0),
        'arm_x': _clamp(38.0 + p[4] * 8.0, 16.0, 70.0),
        'arm_y': _clamp(72.0 + p[5] * 14.0, 30.0, 130.0),
    }


def _advance_ball(x, y, vx, vy):
    """슛 후보별 네트 통과와 착지점을 비교하기 위한 실제 순서의 공 물리."""
    vy = _clamp(vy, -BALL_MAX_Y_SPEED, BALL_MAX_Y_SPEED)
    if x + vx < 0 or x + vx > GROUND_WIDTH:
        vx = -vx
    if y + vy < 0:
        vy = 1.0
    if abs(x - NET_X) < NET_HALF_WIDTH and y > NET_TOP_Y:
        if y <= NET_CAP_BOTTOM_Y:
            if vy > 0:
                vy = -vy
        elif x < NET_X:
            vx = -abs(vx)
        else:
            vx = abs(vx)
    next_y = y + vy
    if next_y > GROUND_Y:
        return x, GROUND_Y, vx, -vy, True
    return x + vx, next_y, vx, vy + 1.0, False


def _project_shot(ball_x, ball_y, ball_vy, shot_y):
    """현재 파워히트 시 (상대 코트 도달, 착지 X, 비행 프레임)을 예측."""
    x, y, vx = ball_x, ball_y, 20.0
    vy = abs(ball_vy) * shot_y * 2.0
    crossed = False
    for frame in range(1, 181):
        x, y, vx, vy, ground = _advance_ball(x, y, vx, vy)
        if x > NET_X:
            crossed = True
        if ground:
            return crossed and x > NET_X, x, frame
    return False, x, 180


def _choose_shot(ball_x, ball_y, ball_vy, self_x, opponent_x):
    """학습된 점수로 위/수평/아래 파워샷 후보를 비교한다."""
    best_shot, best_score = -1, -1.0e100
    for index, shot in enumerate(SHOT_TYPES):
        crosses, landing_x, flight = _project_shot(ball_x, ball_y, ball_vy, shot)
        if not crosses:
            continue
        features = [
            1.0,
            abs(landing_x - opponent_x) / 152.0,
            flight / 60.0,
            (landing_x - 324.0) / 108.0,
            (self_x - 108.0) / 76.0,
        ]
        row = ATTACK_WEIGHTS[6 + index * 5:11 + index * 5]
        score = sum(weight * feature for weight, feature in zip(row, features))
        if score > best_score:
            best_shot, best_score = shot, score
    return best_shot, best_score


def basic_attack_candidate(s):
    """학습된 기본 공격을 공통 후보 딕셔너리 형식으로 반환한다."""
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
    state = int(_number(me.get('state'), 0))
    settings = _settings()

    future_x = _clamp(ball_x + ball_vx * settings['lead'], 32, 184)
    dx = future_x - self_x
    align = 1 if dx > settings['deadband'] else (-1 if dx < -settings['deadband'] else 0)

    if state in (3, 4):
        canonical_action = {'x': align, 'y': 0, 'hit': 0}
        score = 0.2
    elif state == 0:
        jump_now = ball_y <= settings['jump_y'] and ball_vy >= settings['jump_vy']
        canonical_action = {'x': align, 'y': -1 if jump_now else 0, 'hit': 0}
        score = 0.55
    else:
        shot_y, shot_score = _choose_shot(
            ball_x, ball_y, ball_vy, self_x, opponent_x
        )
        close = (
            abs(ball_x - self_x) <= settings['arm_x']
            and abs(ball_y - self_y) <= settings['arm_y']
        )
        canonical_action = {
            'x': 1 if close else align,
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
        'name': 'basic_attack_rl',
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
