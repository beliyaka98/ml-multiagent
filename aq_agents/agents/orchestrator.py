"""Orchestrator — только планирует: понимает запрос и решает, какие агенты и в каком порядке работают.
Предметную работу (данные, прогноз, советы) не выполняет, инструментов у него нет.
"""
from __future__ import annotations

from ..messages import Message, TaskPlan, UserQuery
from ..runtime import AgentContext, BaseAgent, OutputRejected
from ..tools.openmeteo import today_local

AGENT_ORDER = ["data_agent", "forecast_agent", "analyst_agent", "health_agent", "reporter_agent", "critic_agent"]

SYSTEM = """
Ты — Orchestrator многоагентной системы мониторинга качества воздуха в городах Казахстана.
Твоя единственная задача — понять запрос пользователя и составить план: какие агенты и в каком
порядке должны работать. Ты не отвечаешь на вопрос сам, не придумываешь цифр и не анализируешь данные.

Агенты:
- data_agent — собирает данные о загрязнении воздуха и погоде для города;
- forecast_agent — прогноз PM2.5 на сегодня и ближайшие дни (ML-модель + модель CAMS);
- analyst_agent — объясняет причины загрязнения (ветер, инверсия, отопительный сезон);
- health_agent — рекомендации для здоровья с учётом группы людей;
- reporter_agent — пишет итоговый ответ пользователю;
- critic_agent — проверяет итоговый ответ на ошибки и соответствие данным.

Типы запроса (intent):
- forecast — каким будет воздух (сегодня, завтра, на несколько дней);
- current — какой воздух сейчас;
- explain — почему воздух грязный, что на это влияет;
- health — можно ли гулять, бегать, открывать окна и т.п.;
- out_of_scope — запрос не о качестве воздуха (тогда steps = []).

Правила:
1. Если город не указан, используй "Алматы" и отметь это в rationale.
2. Название города пиши по-русски в именительном падеже ("Астана", не "Нур-Султан" и не "в Астане").
3. days_ahead: 0 — сегодня, 1 — завтра, 2 — послезавтра, максимум 3. Для current — 0.
4. user_group: children, elderly, respiratory (астма, болезни лёгких), athletes, иначе general.
5. data_agent нужен всегда, кроме out_of_scope. forecast_agent — для forecast и health,
   а также для explain, если вопрос о будущем. analyst_agent — для explain и при явном вопросе «почему».
   health_agent — для health и если указана группа риска. reporter_agent и critic_agent — всегда, кроме out_of_scope.
"""


class Orchestrator(BaseAgent):
    name = "orchestrator"
    system_prompt = SYSTEM
    decision_model = TaskPlan
    accepts = ("user_query",)

    def parse_input(self, msg: Message, ctx: AgentContext) -> UserQuery:
        return UserQuery.model_validate(msg.payload)

    def prompt(self, inp: UserQuery, ctx: AgentContext) -> str:
        return f"Сегодня {today_local().isoformat()}.\nЗапрос пользователя: {inp.text}"

    def finalize(self, plan: TaskPlan, inp, ctx: AgentContext, facts: dict) -> TaskPlan:
        if plan.intent == "out_of_scope":
            return plan.model_copy(update={"steps": [], "city": None})
        steps = [s for s in AGENT_ORDER if s in set(plan.steps)]  # канонический порядок, без повторов
        if "data_agent" not in steps:
            raise OutputRejected("для запроса о качестве воздуха план должен включать data_agent")
        if plan.intent in ("forecast", "health") and "forecast_agent" not in steps:
            raise OutputRejected(f"для intent={plan.intent} план должен включать forecast_agent")
        city = (plan.city or "Алматы").strip()
        return plan.model_copy(update={"steps": steps, "city": city})
