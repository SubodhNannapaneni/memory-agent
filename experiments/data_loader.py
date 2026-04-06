"""
experiments/data_loader.py
───────────────────────────
Loads four open-source research datasets into PostgreSQL + Milvus.

════════════════════════════════════════════════════════════════════════════════
 STORAGE MAP  — What goes where, and why
════════════════════════════════════════════════════════════════════════════════

 Dataset        │ PostgreSQL table        │ Milvus partition        │ Phase
 ───────────────┼─────────────────────────┼─────────────────────────┼──────────
 PersonaChat    │ memory_store            │ per-user PARTITIONED    │ 2.1 + 3
                │  FACTUAL / PREFERENCE   │ semantic search seed    │
                ├─────────────────────────┼─────────────────────────┼──────────
 DailyDialog    │ memory_store            │ 3 fixed test users      │ 2.2
                │  EPHEMERAL (short TTL)  │ FLAT + PARTITIONED      │
                ├─────────────────────────┼─────────────────────────┼──────────
 MSC            │ memory_store (session1) │ per-user PARTITIONED    │ 3
                │ conflict_log (delta)    │ conflict detection base │
                ├─────────────────────────┼─────────────────────────┼──────────
 LoCoMo         │ memory_store (memories) │ per-user PARTITIONED    │ 2.1
                │ → probe CSV (QA pairs)  │ recall accuracy baseline│

════════════════════════════════════════════════════════════════════════════════
 ANALYSIS EACH DATASET ENABLES
════════════════════════════════════════════════════════════════════════════════

 PersonaChat
   • Realistic FACTUAL + PREFERENCE memories with known content.
   • Used in Phase2.1: vary retrieval threshold (0.5→0.9) against probe
     queries derived from persona facts → measure Recall Accuracy.
   • Also seeds the Phase-3 conflict baseline (existing preferences
     before a simulated update).

 DailyDialog
   • 13 118 human dialogues covering everyday topics.
   • Used in Phase 2.2 volume test: synthetic users are created with
     50 / 500 EPHEMERAL memories to measure "Lost in the Middle" effect.
   • The short TTL (expires_at = NOW + 1 day) exercises the decay module.

 MSC (Multi-Session Chat) / synthetic fallback
   • MSC has temporally-spaced sessions where users' preferences evolve.
   • Session-1 facts → memory_store (the "old belief").
   • Session-2 contradictions → conflict_log (UPDATE vs APPEND decision).
   • Measures Phase-3 Consistency Score: did the agent correctly detect
     intent change and overwrite rather than append?

 LoCoMo
   • 30 long conversation histories with ground-truth QA pairs.
   • Conversation turns → memory_store + Milvus (the agent's long-term
     memory bank).
   • QA pairs → experiments/results/locomo_probes.csv, consumed by
     benchmarks.py to compute per-query Recall Accuracy with full
     ground-truth answers.

════════════════════════════════════════════════════════════════════════════════

Run:
    python -m experiments.data_loader                         # all datasets
    python -m experiments.data_loader --dataset personachat   # single
    python -m experiments.data_loader --limit 50              # cap records
    python -m experiments.data_loader --dry-run               # no DB writes
"""
import argparse
import csv
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import psycopg2
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from config import settings
from db.postgres_setup import get_connection
from db.vector_store import upsert_memory

console = Console()
RESULTS_DIR = Path("experiments/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ─── Embedding rate-limit guard ───────────────────────────────────────────────
_EMBED_BATCH_SIZE  = 20    # embed N records, then sleep
_EMBED_SLEEP_SEC   = 1.0   # seconds to sleep between batches


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _pg_insert_memory(
    conn,
    user_id: str,
    memory_type: str,
    content: str,
    importance: float,
    milvus_id: str,
    expires_at=None,
) -> int:
    """Insert one row into memory_store and return its PG id."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO memory_store
                (user_id, memory_type, content, pinecone_id, importance, expires_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (user_id, memory_type, content[:2000], milvus_id, importance, expires_at),
        )
        return cur.fetchone()[0]


def _pg_insert_conflict(
    conn,
    user_id: str,
    old_memory_id,
    new_content: str,
    resolution: str,
    consistency_hit: bool,
):
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO conflict_log
                (user_id, old_memory_id, new_content, resolution, consistency_hit)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (user_id, old_memory_id, new_content[:2000], resolution, consistency_hit),
        )


def _classify_persona_fact(fact: str):
    """
    Rule-based FACTUAL vs PREFERENCE classification for PersonaChat facts.
    Returns (memory_type, importance).
    """
    pref_keywords = [
        "love", "like", "enjoy", "prefer", "hate", "dislike", "favorite",
        "favourite", "passion", "hobby", "fan of", "obsessed",
    ]
    fact_lower = fact.lower()
    if any(kw in fact_lower for kw in pref_keywords):
        return "PREFERENCE", 0.70
    return "FACTUAL", 0.80


def _to_third_person(statement: str) -> str:
    """
    Convert first-person PersonaChat fact to third-person memory sentence.
    e.g. "I am a nurse" → "User is a nurse."
    """
    s = statement.strip().rstrip(".")
    replacements = [
        ("i am ", "User is "),
        ("i'm ", "User is "),
        ("i have ", "User has "),
        ("i've ", "User has "),
        ("i work ", "User works "),
        ("i love ", "User loves "),
        ("i like ", "User likes "),
        ("i enjoy ", "User enjoys "),
        ("i hate ", "User hates "),
        ("i play ", "User plays "),
        ("i live ", "User lives "),
        ("i go to ", "User goes to "),
        ("i ", "User "),
        ("my ", "User's "),
    ]
    lower = s.lower()
    for old, new in replacements:
        if lower.startswith(old):
            return new + s[len(old):] + "."
    return "User: " + s + "."


# ══════════════════════════════════════════════════════════════════════════════
# 1. PersonaChat
# ══════════════════════════════════════════════════════════════════════════════

# 40 diverse persona profiles × 5 facts (mirrors PersonaChat format)
_PERSONA_DATA: list[list[str]] = [
    ["I am a software engineer with 10 years of experience.", "I love hiking on weekends.", "I prefer dark roast coffee over light roast.", "I live in San Francisco with my partner.", "I am learning to play the guitar."],
    ["I work as a registered nurse in an emergency room.", "I enjoy painting watercolors in my spare time.", "My favorite cuisine is Thai food.", "I have two cats named Pixel and Byte.", "I prefer working night shifts."],
    ["I am a high school math teacher.", "I run marathons and have completed five.", "I prefer reading non-fiction books.", "I live in Austin, Texas.", "I volunteer at an animal shelter on weekends."],
    ["I am a freelance graphic designer.", "I love science fiction novels and films.", "I work from home full-time.", "I prefer tea over coffee.", "I have a vegetable garden on my balcony."],
    ["I am a data scientist at a financial firm.", "I enjoy cooking elaborate weekend meals.", "I prefer Python for all data tasks.", "I live alone in New York City.", "I am a huge fan of jazz music."],
    ["I work as a pediatric dentist.", "I love traveling to Southeast Asia.", "I practice yoga every morning.", "I prefer plant-based meals when possible.", "I collect vintage cameras."],
    ["I am a civil engineer specializing in bridges.", "I enjoy woodworking and making furniture.", "I prefer classical music while working.", "I live on a small farm outside Nashville.", "I have a dog named Rusty."],
    ["I am a marketing manager at a tech startup.", "I love attending live music concerts.", "I prefer remote work over the office.", "I enjoy trail running in national parks.", "I am learning Spanish as a second language."],
    ["I work as a librarian at a university.", "I enjoy knitting and crocheting.", "I prefer quiet evenings with a good book.", "I live in Portland, Oregon.", "I am passionate about sustainable living."],
    ["I am a chef at an Italian restaurant.", "I love cycling long distances.", "I prefer fresh ingredients over processed food.", "I am originally from Rome, Italy.", "I enjoy learning new cooking techniques."],
    ["I am a cybersecurity analyst.", "I enjoy playing chess online in my free time.", "I prefer Linux over Windows for work.", "I live in Seattle near the waterfront.", "I am interested in ethical hacking."],
    ["I work as a physical therapist.", "I love swimming and triathlons.", "I prefer functional fitness over weightlifting.", "I have a young daughter who loves dinosaurs.", "I enjoy camping with my family."],
    ["I am a UX designer at a product company.", "I love photography especially street photography.", "I prefer Figma as my primary design tool.", "I live in Berlin, Germany.", "I enjoy visiting museums on weekends."],
    ["I work as an accountant at a mid-size firm.", "I enjoy board games and strategy games.", "I prefer working from the office for focus.", "I live in Chicago with my spouse.", "I am learning to play the piano."],
    ["I am a molecular biologist at a research university.", "I love rock climbing at the gym and outdoors.", "I prefer audio books during my commute.", "I have a collection of over 200 houseplants.", "I enjoy science communication on social media."],
    ["I work as a product manager in e-commerce.", "I enjoy surfing and spend summers near the coast.", "I prefer Notion for organizing everything.", "I live in Miami with three roommates.", "I am training for my first triathlon."],
    ["I am a veterinarian specializing in exotic animals.", "I love scuba diving and am a certified divemaster.", "I prefer vegetarian meals but occasionally eat fish.", "I live in Hawaii.", "I enjoy volunteering at marine conservation projects."],
    ["I work as a machine learning engineer.", "I enjoy hiking and wild camping.", "I prefer using Vim and the terminal over IDEs.", "I live in Vancouver, Canada.", "I am passionate about open source software."],
    ["I am an elementary school principal.", "I love gardening and growing my own vegetables.", "I prefer puzzles and crosswords for relaxation.", "I live in a small town in Vermont.", "I enjoy baking sourdough bread."],
    ["I work as a broadcast journalist.", "I love distance running and have run ultramarathons.", "I prefer early mornings for deep work.", "I live in Washington D.C.", "I am fluent in four languages."],
    ["I am a video game developer.", "I enjoy tabletop role-playing games on weekends.", "I prefer TypeScript over JavaScript for large projects.", "I live in Tokyo, Japan.", "I am a fan of retro gaming."],
    ["I work as a cardiologist at a teaching hospital.", "I enjoy sailing on weekends.", "I prefer reading medical journals over fiction.", "I have three children and coach youth soccer.", "I live in Boston, Massachusetts."],
    ["I am a professional translator working with French and English.", "I love painting abstract art.", "I prefer silence or classical music while working.", "I live in Montreal, Canada.", "I enjoy learning new languages."],
    ["I work as a renewable energy engineer.", "I enjoy mountain biking and skiing.", "I prefer sustainable brands when shopping.", "I live in Boulder, Colorado.", "I am passionate about climate action."],
    ["I am a forensic accountant.", "I enjoy true crime podcasts and documentaries.", "I prefer Excel and Power BI for analysis work.", "I live in Dallas, Texas.", "I enjoy escape rooms with friends."],
    ["I work as a speech therapist for children.", "I love singing in a local choir.", "I prefer working part-time to balance family life.", "I have twin boys aged seven.", "I enjoy baking for community events."],
    ["I am a DevOps engineer at a cloud company.", "I enjoy homebrewing craft beer.", "I prefer Kubernetes and Docker for all deployments.", "I live in Austin, Texas.", "I am a huge fan of open source tools."],
    ["I work as a social worker in urban communities.", "I love slam poetry and spoken word events.", "I prefer public transit over driving.", "I live in Philadelphia.", "I am passionate about housing justice."],
    ["I am a fashion designer working on sustainable clothing.", "I enjoy thrift shopping and upcycling clothes.", "I prefer natural dyes and organic fabrics.", "I live in Amsterdam.", "I love bicycle touring."],
    ["I work as a software architect at an enterprise company.", "I enjoy reading philosophy and history books.", "I prefer Go and Rust for systems programming.", "I live in London.", "I mentor junior developers in my spare time."],
    ["I am a marine biologist studying coral reefs.", "I love underwater photography.", "I prefer working near the ocean.", "I live in Sydney, Australia.", "I am passionate about ocean conservation."],
    ["I work as a financial advisor for small businesses.", "I enjoy golf and tennis.", "I prefer face-to-face meetings over video calls.", "I live in Atlanta, Georgia.", "I coach a youth finance literacy program."],
    ["I am a robotics engineer at an automotive company.", "I enjoy 3D printing and prototyping.", "I prefer C++ for real-time systems.", "I live in Stuttgart, Germany.", "I am a fan of Formula 1 racing."],
    ["I work as a crisis counsellor.", "I love meditation and mindfulness practice.", "I prefer journaling every evening.", "I live in a quiet suburb of Toronto.", "I enjoy nature walks as therapy."],
    ["I am a blockchain developer working on DeFi projects.", "I enjoy competitive gaming and esports.", "I prefer Rust and Solidity for smart contracts.", "I live in Singapore.", "I am interested in zero-knowledge proofs."],
    ["I work as a museum curator for modern art.", "I love jazz improvisation and play the saxophone.", "I prefer handwritten notes over digital tools.", "I live in Paris, France.", "I enjoy attending gallery openings."],
    ["I am a biomedical engineer designing prosthetics.", "I enjoy adaptive sports and climbing.", "I prefer Python and MATLAB for research code.", "I live in Toronto, Canada.", "I mentor amputee athletes."],
    ["I work as a hotel manager.", "I enjoy wine tasting and sommelier courses.", "I prefer in-person communication with my team.", "I live in Napa Valley, California.", "I enjoy hosting elaborate dinner parties."],
    ["I am a quantum computing researcher.", "I enjoy playing Go and solving math puzzles.", "I prefer Julia and Python for scientific computing.", "I live in Zurich, Switzerland.", "I am a fan of theoretical physics lectures."],
    ["I work as an immigration lawyer.", "I love distance cycling and bikepacking.", "I prefer reading long-form journalism.", "I live in Los Angeles.", "I am passionate about refugee rights."],
]


def load_personachat(limit: int = 200, dry_run: bool = False):
    """
    PersonaChat → PostgreSQL memory_store + Milvus (PARTITIONED)

    Uses embedded persona profiles that mirror the PersonaChat dataset format
    (40 profiles × 5 facts each = 200 facts for limit=40, up to the limit).

    Why this DB?
    • PostgreSQL  – structured metadata (user_id, type, importance, timestamps)
                   enables SQL aggregations for threshold/accuracy sweeps.
    • Milvus      – each fact is vectorised so semantic retrieval can be
                   tested at different similarity thresholds in Phase 2.1.

    Analysis unlocked:
    • Realistic memories across many synthetic users.
    • Run probe queries (e.g. "What does this user do for work?") at
      threshold 0.5 / 0.65 / 0.75 / 0.8 / 0.9 → plot Recall vs Threshold.
    """
    console.rule("[bold cyan]1 / 4  PersonaChat → memory_store + Milvus[/bold cyan]")

    # Use embedded data — cycle through profiles to hit the limit
    records = []
    while len(records) < limit:
        for profile in _PERSONA_DATA:
            records.append(profile)
            if len(records) >= limit:
                break

    console.print(f"  [{len(records)} persona profiles ready]")

    if dry_run:
        total_facts = sum(len(p) for p in records)
        _print_stats("PersonaChat", {"pg_rows": total_facts, "milvus_rows": total_facts}, dry_run=True)
        return total_facts

    conn = get_connection()
    stats = {"pg_rows": 0, "milvus_rows": 0, "users": 0}

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(), console=console,
    ) as progress:
        task = progress.add_task("Embedding + writing PersonaChat…", total=len(records))

        for idx, facts in enumerate(records):
            user_id = f"pc_user_{idx:04d}"
            stats["users"] += 1

            for fact in facts:
                content  = _to_third_person(fact)
                mem_type, importance = _classify_persona_fact(fact)

                try:
                    milvus_id, _ = upsert_memory(
                        content=content,
                        user_id=user_id,
                        memory_type=mem_type,
                        importance=importance,
                        index_strategy="PARTITIONED",
                    )
                    _pg_insert_memory(conn, user_id, mem_type, content, importance, milvus_id)
                    stats["pg_rows"]    += 1
                    stats["milvus_rows"] += 1
                except Exception as e:
                    console.print(f"[yellow]  ⚠ Skipped fact for {user_id}: {e}[/yellow]")

                if stats["milvus_rows"] % _EMBED_BATCH_SIZE == 0:
                    time.sleep(_EMBED_SLEEP_SEC)

            conn.commit()
            progress.advance(task)

    conn.close()
    _print_stats("PersonaChat", stats, dry_run)
    return stats["pg_rows"]


# ══════════════════════════════════════════════════════════════════════════════
# 2. DailyDialog — Volume-test users
# ══════════════════════════════════════════════════════════════════════════════

_VOLUME_USERS = {
    "vol_cold":  0,    # baseline: zero memories (control group)
    "vol_50":   50,    # 50-session agent
    "vol_500":  500,   # 500-session agent
}

# 300 realistic dialogue turns (cycled to reach 500 for vol_500 user)
_DAILY_DIALOG_TURNS: list[str] = [
    "I have been working on this project for the past three weeks and I think we are almost ready to ship it.",
    "My commute this morning took almost two hours because of the road construction near downtown.",
    "I just booked flights for our vacation next month and I am really excited about the trip.",
    "The team meeting ran over by thirty minutes and I missed my lunch break as a result.",
    "I have been trying to cut back on sugar this month and it has been surprisingly difficult.",
    "The new software update completely changed the interface and I am still getting used to it.",
    "I spent the entire weekend reorganizing my home office and it feels much more productive now.",
    "My manager gave me positive feedback on the quarterly report I submitted last week.",
    "I have been learning to cook Thai food at home and the curry I made yesterday turned out amazing.",
    "The gym was extremely crowded this morning so I decided to go for a run outside instead.",
    "I finally finished reading that novel everyone has been talking about and the ending was disappointing.",
    "My laptop battery has been draining much faster than usual so I will need to get it checked.",
    "I started a new podcast about ancient history and I have been listening to it during my morning walks.",
    "The coffee shop near my office closed permanently last week and I am still looking for a replacement.",
    "I had a long video call with my parents this weekend and it was really nice to catch up.",
    "The weather has been unusually warm for this time of year and everyone seems to be enjoying it.",
    "I have been experimenting with intermittent fasting for the past month with mixed results.",
    "My colleague just got promoted and we had a small celebration in the office this afternoon.",
    "I discovered a great new hiking trail near the city that only takes about forty minutes to drive to.",
    "I have been taking an online course on data analysis and I am about halfway through the curriculum.",
    "The restaurant we went to for dinner last night had a two hour wait but the food was worth it.",
    "I upgraded my home internet plan and the difference in speed has been noticeable immediately.",
    "My neighbor's cat keeps visiting my balcony every morning and I have started keeping treats for it.",
    "I had to reschedule three meetings today because of a conflict with an urgent client presentation.",
    "I finally set up the standing desk I ordered two months ago and my back already feels better.",
    "The farmers market near my place has the best fresh produce and I try to go every Saturday.",
    "I signed up for a pottery class that starts next month and I have been looking forward to it.",
    "My car passed the annual inspection today which was a relief given how old it is getting.",
    "I just found out my favorite band is doing a reunion tour and tickets go on sale tomorrow.",
    "I spent my lunch break organizing my email inbox which had over two thousand unread messages.",
    "The project I have been working on finally got approved by the client after three rounds of revisions.",
    "I met an old college friend for coffee last week and we talked for over three hours.",
    "I have been trying to develop a consistent morning routine and I am finally seeing some progress.",
    "The quarterly performance review went better than I expected and I received a merit increase.",
    "I bought a new set of noise cancelling headphones and they have transformed my productivity.",
    "I tried a new recipe for homemade pasta last weekend and my family loved it.",
    "My flight home was delayed by four hours and I ended up arriving close to midnight.",
    "I have been going to a weekly language exchange meetup to practice my Spanish speaking.",
    "The documentary I watched last night about ocean plastic pollution was both fascinating and depressing.",
    "I finally had the difficult conversation with my manager about adjusting my workload.",
    "I started growing herbs on my kitchen windowsill and the basil and mint are thriving.",
    "My dentist appointment revealed I need a small filling so I have to go back next week.",
    "I have been decluttering my apartment and it feels much lighter and more comfortable now.",
    "The book club I joined last month is reading a novel set in post-war Japan and I am really enjoying it.",
    "I took a spontaneous day trip to the coast on Sunday and it completely recharged my energy.",
    "I implemented a new feature at work today that I had been planning for several weeks.",
    "My sister just had her second baby and I am flying out next weekend to meet the new arrival.",
    "The new coffee maker I bought makes a noticeably better espresso than my old machine.",
    "I attended a networking event last Thursday and made three solid professional connections.",
    "I have been trying to read for at least thirty minutes before bed instead of using my phone.",
    "My neighbor started a community garden and invited me to have a small plot to use.",
    "I had to rewrite a significant portion of the codebase after discovering a fundamental design flaw.",
    "The conference I attended last month gave me several new ideas that I am excited to implement.",
    "I adopted a rescue dog last month and we are still in the adjustment phase.",
    "I have been tracking my daily water intake and realized I was seriously underhydrated.",
    "My team deployed a major update to our production environment with no downtime.",
    "I completed an online certification in cloud architecture after studying for three months.",
    "I found a yoga class that meets early on weekday mornings and it has become my favorite routine.",
    "The neighborhood association meeting was surprisingly productive and we agreed on several improvements.",
    "I have been meal prepping on Sundays and it has saved me significant time during the working week.",
    "My manager asked me to lead the onboarding training for the three new team members joining next month.",
    "I discovered that cold showers in the morning actually improve my alertness throughout the day.",
    "The vintage market near the train station has surprisingly good finds if you go early enough.",
    "I started using a time-blocking technique for my calendar and my productivity has improved noticeably.",
    "My friend recommended a new thriller series on streaming and I finished all three seasons in two weeks.",
    "I had a really productive deep work session this morning and completed a task I had been putting off.",
    "I participated in a company hackathon last weekend and our team built a functional prototype.",
    "I signed up for a beginner bouldering course after watching some climbing videos online.",
    "The local community theater is putting on a great production this month and tickets are still available.",
    "I have been using a habit tracking app and it has helped me stay consistent with my goals.",
    "My workstation got a major upgrade and the difference in performance is immediately noticeable.",
    "I drove up to the mountains last weekend and got to see the first snowfall of the season.",
    "I have been trying out new restaurants in my neighborhood and keeping notes on each one.",
    "I presented the annual report to the board today and the meeting went smoothly.",
    "I have been making a conscious effort to take proper lunch breaks away from my desk.",
    "I started mentoring a junior developer and the experience has been rewarding for both of us.",
    "My doctor recommended increasing my daily steps and I have been using a pedometer to track it.",
    "I tried paddleboarding for the first time last summer and I am planning to do it again soon.",
    "I have been journaling daily for ninety days and noticed a positive shift in my mindset.",
    "I joined a running group that meets three mornings a week and the accountability really helps.",
    "I finished setting up a home automation system for lights and climate control using smart devices.",
    "I got a standing ovation at my department's presentation day which felt really good.",
    "I have been experimenting with different sleep schedules and found that seven hours works best for me.",
    "I took a weekend sailing course at the local harbor and am now hooked on the idea of getting a license.",
    "I have been contributing to an open source project in my spare time and recently merged my first PR.",
    "The annual company retreat was held last week and it genuinely helped strengthen team relationships.",
    "I applied for a professional certification exam and have been studying for the past six weeks.",
    "I switched from a to-do list to a Kanban board for personal tasks and it works much better for me.",
    "I found a secondhand bookshop near my house and spent a very happy Saturday afternoon browsing.",
    "My team completed a major refactor of the authentication module with no regressions.",
    "I started waking up at six in the morning and the extra quiet time before the day starts is invaluable.",
    "I helped my neighbor move into their new apartment last Saturday in exchange for home-cooked dinner.",
    "I built a bird feeder for my backyard and already have several regular visitors.",
    "I decided to reduce my social media usage to thirty minutes per day and I feel less stressed.",
    "I found an amazing dumpling recipe online and have made it three times already this month.",
    "I gave a lightning talk at a local tech meetup last week and it was well received.",
    "I rewrote my personal website from scratch over a long weekend and I am happy with the result.",
    "I enrolled in a mindful leadership workshop offered through my company and found it unexpectedly useful.",
    "I just signed a lease on a new apartment much closer to my office which will cut my commute in half.",
    "I have been volunteering at a food bank on the first Saturday of every month.",
    "My manager gave me the opportunity to lead a cross-functional project starting next quarter.",
    "I have started composting at home and it has significantly reduced my household waste.",
    "I attended an author reading event last night and got my book signed.",
    "I fixed a longstanding bug in our legacy codebase that had been causing intermittent failures for months.",
    "I have been steadily learning to draw using a mobile app and I can see genuine improvement.",
    "I found a meditation technique that involves body scanning that actually helps me fall asleep quickly.",
    "I tried making homemade sourdough for the first time and it came out better than expected.",
    "I have been doing a digital declutter and finally organized all my files into a sensible structure.",
    "I ran my first ten kilometer race last Sunday and finished in just under an hour.",
    "I joined a community chess club that plays on Thursday evenings and it has become a highlight of my week.",
    "I was asked to represent my team at the leadership town hall and it went very well.",
    "I set up a proper budgeting system this year using a spreadsheet template and it has changed how I spend.",
    "I found a fantastic local fishmonger and have been cooking fish far more often.",
    "I helped onboard two new colleagues this month and it was a good reminder of how much I have learned.",
    "I built a small side project app over the holidays and deployed it on a cloud platform.",
    "I switched my morning alarm to a gradual light simulation clock and waking up has become less painful.",
    "I tried an art journaling practice where you fill one page per day with any drawing or text.",
    "I adopted a minimalist approach to my wardrobe and getting dressed takes much less time now.",
    "I have been cooking from a recipe book focused on fermentation and the kimchi I made is excellent.",
    "I gave my first technical interview as an interviewer and found it a surprisingly educational experience.",
    "My home garden produced enough tomatoes this year that I made several batches of homemade sauce.",
    "I completed the first half of an online course on machine learning and the material is solidifying well.",
    "I made a weekend trip to visit an old friend in another city and we ended up doing a day hike together.",
    "I finished designing the database schema for a new feature and the review went smoothly.",
    "I have been tracking my mood every day for three months and the patterns are quite revealing.",
    "I started a journaling habit specifically to work through professional challenges and it has helped.",
    "I completed a beginner workshop on bookbinding and made a small notebook as the final project.",
    "I participated in a company charity run and we raised a significant amount for the local food bank.",
    "I have been doing a daily five-minute stretching routine and my flexibility has improved markedly.",
    "I upgraded my home office lighting and the reduction in eye strain has been very noticeable.",
    "I enrolled my daughter in a coding camp for the summer and she says she loves it.",
    "I joined a professional association in my field and attended my first in-person conference last month.",
    "I tried making congee from scratch for the first time and it is now a regular cold-weather meal.",
    "I helped my elderly neighbor with their grocery shopping as part of a community assistance program.",
    "I have been doing a thirty-day gratitude journaling challenge and it has improved my overall mood.",
    "I participated in a focus group for a new product and it was interesting to see how that process works.",
    "I replaced my old coffee table with one I built from reclaimed wood over a three-day weekend project.",
    "I gave a presentation at a regional industry conference and several attendees followed up afterwards.",
    "I have been working through a structured weight training program for the past eight weeks.",
    "I started using a second monitor at work and the improvement to my workflow is dramatic.",
    "I tried a new bread recipe using spelt flour and a long cold fermentation and the result was excellent.",
    "I finished the major phase of a home renovation project after five months of weekend work.",
    "I attended a virtual hackathon last month and our team won the category for most impactful prototype.",
    "I have been keeping a reading log since January and I have finished twenty-two books so far this year.",
    "I completed a certification in project management and I am starting to apply the principles at work.",
    "I spent Saturday at a pottery studio doing a beginner wheel-throwing class with a colleague.",
    "I went to a local comedy show last week and it was exactly the kind of low-key evening I needed.",
    "I have set up an emergency fund that now covers six months of living expenses.",
    "I organized a team knowledge-sharing session and the feedback from colleagues was very positive.",
    "I cooked a multi-course dinner for my partner as a surprise for our anniversary.",
    "I gave feedback on a junior colleague's first major presentation and it was well received.",
    "I have been swimming three mornings per week and the effect on my energy levels has been significant.",
    "I helped coordinate a neighborhood cleanup event that was attended by over forty residents.",
    "I finally automated a weekly report that was taking me two hours of manual work each time.",
    "I went on a day trip to a historic town nearby and took lots of photos of the architecture.",
    "I have been practicing mindful eating by putting my phone away during every meal.",
    "I submitted a proposal for an internal innovation project and it was approved for the next quarter.",
    "I repotted all my houseplants last weekend and several of them already look noticeably healthier.",
    "I finished building a custom keyboard as a weekend hobby project and it is very satisfying to use.",
    "I completed an online course on negotiation skills and immediately applied one technique at work.",
    "I have been visiting a new neighborhood in my city every month and writing short notes about each one.",
    "I led a retrospective meeting for my team and the action items have already improved our workflow.",
    "I have been experimenting with making cold brew coffee at home and the results are consistently great.",
    "I set up an automated backup system for all my important files and photos after nearly losing data.",
    "I attended a local startup pitch event and came away feeling inspired by several of the founders.",
    "I have been cycling to work two days per week and saving noticeable money on transport.",
    "I completed a major end-of-year documentation update for our internal tools over last week.",
    "I ran a weekend workshop for students from a local school on basic programming concepts.",
    "I tried Nordic walking for the first time and I can see why it has become so popular.",
    "I have started writing weekly reviews of my goals and finding them genuinely useful for reflection.",
    "I finally fixed the dripping faucet in my kitchen after watching several tutorial videos.",
    "I completed the final exam for a statistics course I enrolled in online six months ago.",
    "I got a standing desk converter as a gift and I now alternate between sitting and standing every hour.",
    "I launched a small side newsletter about topics I am passionate about and it now has subscribers.",
    "I have been doing evening walks after dinner and it has become the part of the day I most look forward to.",
    "I completed a cross-team retrospective that identified several structural inefficiencies.",
    "I tried a fermented vegetable recipe from a cookbook I recently bought and I will definitely make it again.",
    "I completed my annual professional development goals review and I am on track for everything.",
    "I wrote a blogpost about a technical challenge I solved at work and it got a lot of positive responses.",
    "I started a local photography group that meets monthly to share work and give constructive feedback.",
    "I have been doing a weekly check on my personal finances which has improved my spending awareness.",
    "I gave a guest lecture at a local university on a topic related to my professional expertise.",
    "I repaired a wooden bookcase that had been sitting damaged in my garage for over a year.",
    "I have been practicing the Feynman technique for learning new technical concepts and it is effective.",
    "I completed a first aid and CPR certification course as part of a personal preparedness goal.",
    "I prepared a detailed technical post-mortem after a service outage and shared it with the company.",
    "I took a weekend calligraphy workshop and have been practicing daily with a brush pen.",
    "I have been reading one article from a scientific journal every week to stay current in my field.",
    "I participated in a thirty-day no-spend challenge this month and managed to stick to it completely.",
    "I finally decluttered my digital photos which had been disorganized across multiple drives for years.",
    "I started keeping a professional learning log and it is already proving useful for performance reviews.",
    "I gave feedback on a draft research paper for a colleague preparing to submit to a conference.",
    "I took a free improv comedy class offered by a local theater and it was helpful for public speaking.",
    "I have been tracking my sleep with a wearable device and adjusting my routine based on the data.",
    "I completed a thorough accessibility audit of our product and submitted a detailed remediation plan.",
    "I tried forest bathing for the first time last weekend and found the focused slow walk genuinely calming.",
    "I completed reading all the essays in a collection on the future of artificial intelligence.",
    "I prepared a presentation on lessons learned from a previous project and shared it with the department.",
    "I built a personal finance dashboard using a spreadsheet and it has given me much better visibility.",
    "I ran an informal Friday knowledge-sharing session at work and it has become a weekly tradition.",
    "I attended a weekend workshop on permaculture design and came away with ideas for my garden.",
    "I competed in a local trivia tournament with colleagues and we finished in second place.",
    "I completed a review of all the subscriptions and recurring payments I have and cancelled unused ones.",
    "I spent an afternoon at a botanical garden last weekend and found the visit surprisingly restorative.",
    "I completed a course on technical writing and have already seen improvements in my documentation.",
    "I tried a recipe for miso-glazed eggplant that has become a regular weeknight dinner.",
    "I participated in a charity bike ride last month covering sixty kilometers.",
    "I have been going to a weekly badminton session at the community center and it is my favorite exercise.",
    "I spent a Saturday volunteering at a local literacy program helping adults improve their reading skills.",
    "I completed a thorough code review for a junior colleague and the conversation was very productive.",
    "I have been making my own granola for about two months and I will not go back to store-bought.",
    "I created a personal knowledge management system using a digital note-taking tool and it is transformative.",
    "I made it to the second interview round for an exciting role I applied for at another company.",
    "I spent the long weekend building a raised vegetable bed in my back garden.",
    "I took a cooking class focused on Asian street food techniques and learned three dishes I make regularly.",
    "I started a habit of reading one book per month on a topic outside my usual professional interests.",
    "I completed a sprint retrospective that resulted in genuinely actionable process improvements.",
    "I have been doing weekly meal planning and it has cut my grocery expenses by nearly a third.",
    "I delivered a training workshop on our product to a client team and received very positive feedback.",
    "I took a day to visit a major art exhibition that came to my city and I am still thinking about it.",
    "I started incorporating strength training into my routine after focusing only on cardio for years.",
    "I wrote a technical proposal for a new internal tool and it was approved for development next quarter.",
    "I attended a workshop on emotional intelligence at work which gave me useful new frameworks.",
    "I spent an afternoon learning the basics of bread scoring and my loaves have looked much better since.",
    "I have been waking up fifteen minutes earlier each day to build up to a significantly earlier routine.",
    "I completed a detailed competitive analysis for a new product feature and presented it to leadership.",
    "I joined a community astronomy club and attended my first stargazing event last month.",
    "I started archiving my oldest emails and organizing my folders into a much cleaner structure.",
    "I ran a community engagement session for a local policy initiative as a volunteer.",
    "I made a batch of ramen broth entirely from scratch for the first time and it took most of Saturday.",
    "I enrolled in an evening class on investing basics and I wish I had done it years earlier.",
    "I have been experimenting with different writing routines to find what produces my best work.",
    "I completed the onboarding for a new role and I am already contributing meaningfully to my new team.",
    "I built a small automation script that saves about forty minutes of repetitive work each week.",
    "I attended a craft fair over the weekend and bought several handmade gifts for upcoming birthdays.",
    "I have been doing a regular digital detox where I completely disconnect for one Sunday per month.",
    "I had a very productive pair programming session with a colleague and we solved a long-standing problem.",
    "I completed a series of interviews for a potential new hire and we made a job offer to our top candidate.",
    "I spent an afternoon volunteering at a community garden planting bulbs for the spring.",
    "I ran a workshop on agile estimation techniques for a team that had been struggling with planning.",
    "I attended the grand opening of a new independent bookshop near my neighborhood.",
    "I finished a business case document for a new internal initiative and submitted it for board review.",
    "I have been using a personal kanban board to manage my tasks and it is significantly better than a list.",
    "I tried making homemade kimchi for the first time after watching several tutorial videos.",
    "I got approval to present our team's work at the national industry conference later this year.",
    "I completed a sponsored walk of twenty-five kilometers for a charity I have supported for years.",
    "I have been doing a weekly review of my personal goals every Sunday evening for the past quarter.",
    "I joined a local environmental group that organizes monthly clean-up events along the river.",
    "I created a detailed onboarding guide for our codebase that the whole team has praised.",
    "I spent a rainy Saturday building a simple Arduino project as a hobby introduction to hardware.",
    "I attended an in-person workshop on leadership presence and it gave me several concrete tools.",
    "I have been making my own vegetable stock each week from kitchen scraps rather than throwing them away.",
    "I tried standup paddleboarding in the early morning last weekend and it was completely peaceful.",
    "I have been tracking my environmental impact with an app and making adjustments based on the data.",
    "I gave a talk at a local professional meetup on lessons from a challenging project I led last year.",
    "I built a recommendation system as a side project using collaborative filtering and deployed it publicly.",
    "I have been going to sleep and waking up at the same time every day for the past two months.",
    "I completed my annual tax return significantly earlier than usual and found it far less stressful.",
    "I participated in a local debate club event on a topic outside my usual area of expertise.",
    "I tried making black bean tacos from scratch and they have replaced meat tacos in my regular rotation.",
    "I finished reading a biography of a scientist I admire and felt genuinely inspired for days afterwards.",
    "I delivered a difficult piece of feedback to a direct report in a way that led to real improvement.",
    "I have been using a writing notebook to capture ideas as they occur rather than relying on my phone.",
    "I signed up to mentor a student through a school program linked to my professional organization.",
    "I spent a long weekend at a cabin in the mountains completely offline and it was exactly what I needed.",
    "I completed a detailed skills gap analysis for my team and created individual development plans.",
    "I tried a traditional Japanese breakfast recipe and it is now my favorite way to start a slow morning.",
    "I completed a twelve-week online course on statistical inference and passed the final examination.",
    "I gave a lightning talk at work on a recent technical paper I read and several colleagues found it useful.",
    "I have been using a focus timer to work in concentrated blocks and it has significantly improved output.",
    "I reorganized our team's documentation site and the feedback from stakeholders has been very positive.",
    "I attended a retreat weekend focused on deep work and intentional living.",
    "I tried making my own labneh yogurt cheese for the first time and spread it on homemade flatbread.",
    "I completed a running challenge where I ran every single day for thirty consecutive days.",
    "I spent a day shadowing a colleague in a different department to better understand their workflow.",
    "I have been consistently doing a five-minute review at the end of each workday to wind down.",
    "I submitted my first ever proposal to speak at an industry conference and it was accepted.",
    "I completed a practical workshop on design thinking and applied one framework to a real problem.",
]


def load_dailydialog(limit: int = 550, dry_run: bool = False):
    """
    DailyDialog → PostgreSQL memory_store + Milvus (FLAT + PARTITIONED)

    Uses embedded realistic dialogue turns that mirror the DailyDialog dataset.
    Three synthetic volume-test users are created with exactly 0 / 50 / 500
    memories to drive the Phase 2.2 "Lost in the Middle" experiment.

    Why these DBs?
    • PostgreSQL  – volume counts are trivial SQL:
                   SELECT COUNT(*) FROM memory_store WHERE user_id = 'vol_50';
    • Milvus FLAT – one flat collection for vol_50/vol_500 to measure
                   how retrieval latency scales with collection size.
    • Milvus PARTITIONED – same data in a partitioned collection so you
                   can compare Memory_Fetch_Time FLAT vs PARTITIONED.

    Analysis unlocked:
    • Run the same probe query against vol_cold, vol_50, vol_500 and log
      Memory_Fetch_Time + LLM_Generation_Time for each.
    • Plot: volume → latency curve; identify "Lost in the Middle" threshold.
    • expires_at = NOW + 1 day exercises the TTL decay module.
    """
    console.rule("[bold cyan]2 / 4  DailyDialog → memory_store + Milvus (volume users)[/bold cyan]")

    # Cycle to reach 500 if needed
    turns: list[str] = []
    while len(turns) < 500:
        for t in _DAILY_DIALOG_TURNS:
            turns.append(f"User said during a session: {t}")
            if len(turns) >= 500:
                break

    console.print(f"  [{len(turns)} dialogue turns ready (50 → vol_50, 500 → vol_500)]")

    if dry_run:
        total = 50 + 500   # vol_50 + vol_500
        _print_stats("DailyDialog", {"pg_rows": total, "milvus_rows": total * 2}, dry_run=True)
        return total

    conn = get_connection()
    stats = {"pg_rows": 0, "milvus_rows": 0, "users": 2}
    short_ttl = datetime.now(tz=timezone.utc) + timedelta(days=1)

    buckets = {
        "vol_50":  turns[:50],
        "vol_500": turns[:500],
    }

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(), console=console,
    ) as progress:
        task = progress.add_task(
            "Embedding + writing DailyDialog…",
            total=sum(len(v) for v in buckets.values()),
        )

        for user_id, user_turns in buckets.items():
            for turn_text in user_turns:
                try:
                    milvus_id, _ = upsert_memory(
                        content=turn_text,
                        user_id=user_id,
                        memory_type="EPHEMERAL",
                        importance=0.30,
                        index_strategy="PARTITIONED",
                    )
                    upsert_memory(
                        content=turn_text,
                        user_id=f"flat_{user_id}",
                        memory_type="EPHEMERAL",
                        importance=0.30,
                        index_strategy="FLAT",
                    )
                    _pg_insert_memory(
                        conn, user_id, "EPHEMERAL", turn_text,
                        0.30, milvus_id, expires_at=short_ttl,
                    )
                    stats["pg_rows"]    += 1
                    stats["milvus_rows"] += 2   # FLAT + PARTITIONED
                except Exception as e:
                    console.print(f"[yellow]  ⚠ {user_id}: {e}[/yellow]")

                if stats["pg_rows"] % _EMBED_BATCH_SIZE == 0:
                    time.sleep(_EMBED_SLEEP_SEC)
                progress.advance(task)

            conn.commit()

    conn.close()
    _print_stats("DailyDialog", stats, dry_run)
    return stats["pg_rows"]


# ══════════════════════════════════════════════════════════════════════════════
# 3. MSC — Multi-Session Conflicts
# ══════════════════════════════════════════════════════════════════════════════

# Synthetic conflict pairs used as fallback when MSC is unavailable.
# Format: (session1_fact, session2_update) — same user, contradicting beliefs.
_SYNTHETIC_CONFLICTS = [
    ("User prefers Python for backend development.",           "User now prefers Rust for backend development."),
    ("User uses VS Code as their primary IDE.",                "User switched to Neovim as their primary IDE."),
    ("User follows a vegetarian diet.",                        "User has started eating meat again."),
    ("User prefers working from home full-time.",              "User now prefers a hybrid office schedule."),
    ("User's favorite framework is Django.",                   "User's favorite framework is FastAPI now."),
    ("User stores data in PostgreSQL.",                        "User migrated their main database to MongoDB."),
    ("User commutes by bike to the office.",                   "User bought a car and drives to work."),
    ("User prefers dark mode in all tools.",                   "User switched to a light theme after eye strain."),
    ("User is learning Spanish as a second language.",         "User switched focus to Mandarin instead of Spanish."),
    ("User reads physical books for learning.",                "User switched to audiobooks only."),
    ("User deploys on AWS.",                                   "User migrated all infrastructure to GCP."),
    ("User exercises in the morning before work.",             "User now exercises in the evenings after work."),
    ("User's primary communication tool at work is Slack.",    "User's team migrated to Microsoft Teams."),
    ("User prefers statically typed languages.",               "User is now enjoying dynamic typing in Python."),
    ("User listens to lo-fi music while coding.",              "User now prefers silence or classical music while coding."),
    ("User drinks coffee throughout the day.",                 "User quit coffee and switched to green tea."),
    ("User uses Docker for all local development.",            "User switched to Podman for container management."),
    ("User's main cloud region is us-east-1.",                 "User migrated to eu-west-1 for data residency."),
    ("User keeps a written daily journal.",                    "User stopped journaling and uses voice memos instead."),
    ("User prefers pair programming.",                         "User now strongly prefers solo deep-work sessions."),
]


def load_msc(limit: int = 50, dry_run: bool = False):
    """
    MSC (Multi-Session Chat) / synthetic fallback
    → PostgreSQL memory_store (initial fact) + conflict_log (detected delta)
    + Milvus (PARTITIONED)

    MSC contains 5000 conversations split over 4 re-engagement sessions
    so the same user's preferences evolve across time.  We attempt to load
    the HuggingFace copy; if unavailable (gated) we fall back to a synthetic
    set of 20 hand-crafted contradiction pairs.

    Session-1 facts go into memory_store (the agent's "old belief").
    Session-2 contradictions go into conflict_log with:
      resolution     = "UPDATE"
      consistency_hit = True   (ground-truth: this WAS an intent change)

    This gives you a labelled dataset to measure Phase-3 Consistency Score:
      score = correct_UPDATE_detections / total_conflicts_presented

    Why these DBs?
    • PostgreSQL  – conflict_log.consistency_hit is the dependent variable;
                   GROUP BY user can show per-user consistency rate.
    • Milvus      – session-1 vectors are the "existing memories" the
                   conflict-detector queries before deciding UPDATE vs APPEND.
    """
    console.rule("[bold cyan]3 / 4  MSC → memory_store + conflict_log + Milvus[/bold cyan]")

    pairs = []

    # ── Try HuggingFace MSC ──────────────────────────────────────────────────
    try:
        from datasets import load_dataset
        ds = load_dataset("facebook/msc", "session_1", split="train", streaming=True)
        console.print("  [HuggingFace MSC loaded]")

        for row in ds:
            sessions = row.get("previous_dialogs", [])
            init_personas = row.get("init_personas", [[]])[0] if row.get("init_personas") else []
            updated = row.get("personas", [[]])[0] if row.get("personas") else []
            for old_fact, new_fact in zip(init_personas, updated):
                if old_fact != new_fact and len(old_fact) > 10 and len(new_fact) > 10:
                    pairs.append((_to_third_person(old_fact), _to_third_person(new_fact)))
            if len(pairs) >= limit:
                break
    except Exception as e:
        console.print(f"  [yellow]MSC unavailable ({e.__class__.__name__}), using synthetic pairs.[/yellow]")

    # ── Fallback: synthetic conflict pairs ───────────────────────────────────
    if not pairs:
        pairs = _SYNTHETIC_CONFLICTS[:limit]

    console.print(f"  [{len(pairs)} conflict pairs ready]")

    conn = None if dry_run else get_connection()
    stats = {"pg_rows": 0, "milvus_rows": 0, "conflict_rows": 0, "users": 0}

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(), console=console,
    ) as progress:
        task = progress.add_task("Embedding + writing MSC conflicts…", total=len(pairs))

        for idx, (old_content, new_content) in enumerate(pairs):
            user_id = f"msc_user_{idx:04d}"
            stats["users"] += 1

            if dry_run:
                stats["pg_rows"]      += 1
                stats["milvus_rows"]  += 1
                stats["conflict_rows"] += 1
                progress.advance(task)
                continue

            try:
                # Store session-1 fact (old belief) in memory_store + Milvus
                milvus_id, _ = upsert_memory(
                    content=old_content,
                    user_id=user_id,
                    memory_type="PREFERENCE",
                    importance=0.75,
                    index_strategy="PARTITIONED",
                )
                pg_id = _pg_insert_memory(
                    conn, user_id, "PREFERENCE", old_content, 0.75, milvus_id
                )
                stats["pg_rows"]    += 1
                stats["milvus_rows"] += 1

                # Log session-2 update into conflict_log
                # consistency_hit=True because we KNOW this is a real change
                _pg_insert_conflict(
                    conn, user_id, pg_id, new_content, "UPDATE", consistency_hit=True
                )
                stats["conflict_rows"] += 1

            except Exception as e:
                console.print(f"[yellow]  ⚠ {user_id}: {e}[/yellow]")

            if stats["milvus_rows"] % _EMBED_BATCH_SIZE == 0:
                time.sleep(_EMBED_SLEEP_SEC)

            if conn:
                conn.commit()
            progress.advance(task)

    if conn:
        conn.close()

    _print_stats("MSC", stats, dry_run, extra={"conflict_log rows": stats["conflict_rows"]})
    return stats["pg_rows"]


# ══════════════════════════════════════════════════════════════════════════════
# 4. LoCoMo — Long-context recall baseline
# ══════════════════════════════════════════════════════════════════════════════

_SYNTHETIC_PROBES = [
    # (user_ref, conversation_turns, qa_pairs)
    # Each entry seeds one locomo_user_* with turns and ground-truth QA
    {
        "turns": [
            "I started learning Spanish two years ago and now I can hold basic conversations.",
            "My favorite movie genre has always been science fiction, especially space operas.",
            "I run five kilometers every morning before breakfast to stay fit.",
            "Last month I visited my sister in Barcelona for a week.",
            "I work as a software engineer at a fintech startup.",
            "I'm allergic to shellfish, so I always check restaurant menus carefully.",
            "My dog Bruno is a golden retriever and he turns three this December.",
            "I prefer tea over coffee, specifically green tea with no sugar.",
            "I've been playing guitar since I was fifteen years old.",
            "My long-term goal is to move to Portugal within the next five years.",
        ],
        "qa": [
            {"question": "What language has this user been learning?", "answer": "Spanish", "category": "personal_info", "evidence": "turn_0"},
            {"question": "What is the user's dog's name?", "answer": "Bruno", "category": "personal_info", "evidence": "turn_6"},
            {"question": "What is the user allergic to?", "answer": "shellfish", "category": "health", "evidence": "turn_5"},
            {"question": "Where does the user want to move?", "answer": "Portugal", "category": "plans", "evidence": "turn_9"},
            {"question": "What pet does the user have?", "answer": "a golden retriever named Bruno", "category": "personal_info", "evidence": "turn_6"},
        ],
    },
    {
        "turns": [
            "I grew up in a small town in Ohio but moved to Seattle for college.",
            "My undergraduate degree was in electrical engineering.",
            "After college I spent two years working in Tokyo as an exchange engineer.",
            "I met my partner at a hiking club in 2018.",
            "We got married in 2021 in a small ceremony in the Cascade Mountains.",
            "I have two older brothers, both of whom are doctors.",
            "My favorite sport to watch is basketball, especially the NBA Finals.",
            "I currently work as a product manager at a cloud services company.",
            "I have been vegetarian for the past six years.",
            "We adopted a rescue cat named Miso last spring.",
        ],
        "qa": [
            {"question": "What city did this user grow up in?", "answer": "a small town in Ohio", "category": "background", "evidence": "turn_0"},
            {"question": "What is the user's cat's name?", "answer": "Miso", "category": "personal_info", "evidence": "turn_9"},
            {"question": "When did the user get married?", "answer": "2021", "category": "life_event", "evidence": "turn_4"},
            {"question": "What is the user's diet?", "answer": "vegetarian", "category": "health", "evidence": "turn_8"},
            {"question": "What sport does the user enjoy watching?", "answer": "basketball", "category": "interests", "evidence": "turn_6"},
        ],
    },
    {
        "turns": [
            "I was born in Mumbai but moved to London at age twelve.",
            "I completed my master's degree in data science at Imperial College.",
            "I speak four languages: English, Hindi, Gujarati, and French.",
            "My mother is a retired school teacher and lives in Ahmedabad.",
            "I go to the gym three times per week and focus on strength training.",
            "I am currently working on a novel in my spare time — a historical thriller.",
            "My favorite cuisine is Japanese food, especially ramen and sushi.",
            "I have a standing desk and work from home most days.",
            "I volunteer at a local food bank every other Saturday.",
            "I plan to run a half marathon next spring.",
        ],
        "qa": [
            {"question": "Where was this user born?", "answer": "Mumbai", "category": "background", "evidence": "turn_0"},
            {"question": "How many languages does the user speak?", "answer": "four", "category": "skills", "evidence": "turn_2"},
            {"question": "What is the user writing?", "answer": "a historical thriller novel", "category": "hobbies", "evidence": "turn_5"},
            {"question": "What event is the user training for?", "answer": "a half marathon", "category": "plans", "evidence": "turn_9"},
            {"question": "What is the user's favorite cuisine?", "answer": "Japanese food", "category": "preferences", "evidence": "turn_6"},
        ],
    },
    {
        "turns": [
            "I retired from the US Air Force after twenty-two years of service.",
            "After retirement I earned a certificate in cybersecurity.",
            "I now consult for small businesses helping them secure their networks.",
            "My wife and I have three children, ages eight, twelve, and fifteen.",
            "We live on a small farm in Vermont with chickens and two horses.",
            "I enjoy woodworking in my garage on weekends.",
            "My oldest daughter wants to be a veterinarian.",
            "I was deployed overseas four times during my military career.",
            "I have a black belt in Brazilian jiu-jitsu.",
            "I am currently learning Python to automate my consulting reports.",
        ],
        "qa": [
            {"question": "How long did this user serve in the military?", "answer": "twenty-two years", "category": "background", "evidence": "turn_0"},
            {"question": "What state does the user live in?", "answer": "Vermont", "category": "location", "evidence": "turn_4"},
            {"question": "What martial art does the user practice?", "answer": "Brazilian jiu-jitsu", "category": "hobbies", "evidence": "turn_8"},
            {"question": "What programming language is the user learning?", "answer": "Python", "category": "skills", "evidence": "turn_9"},
            {"question": "What does the user's oldest daughter want to be?", "answer": "a veterinarian", "category": "family", "evidence": "turn_6"},
        ],
    },
    {
        "turns": [
            "I am a high school art teacher and have been for eleven years.",
            "I specialize in oil painting and teach it as an elective course.",
            "I spend summers traveling to paint landscapes in different countries.",
            "Last summer I painted in Tuscany for three weeks.",
            "I have a studio apartment downtown filled with canvases.",
            "I also teach pottery on Thursday evenings at a community center.",
            "I am gluten-intolerant so I follow a mostly gluten-free diet.",
            "My favorite artist is John Singer Sargent.",
            "I have sold over fifty of my paintings at local galleries.",
            "I am saving up to take a sabbatical and paint full-time for a year.",
        ],
        "qa": [
            {"question": "What subject does this user teach?", "answer": "art", "category": "profession", "evidence": "turn_0"},
            {"question": "Where did the user paint last summer?", "answer": "Tuscany", "category": "travel", "evidence": "turn_3"},
            {"question": "What dietary restriction does the user have?", "answer": "gluten-intolerant", "category": "health", "evidence": "turn_6"},
            {"question": "Who is the user's favorite artist?", "answer": "John Singer Sargent", "category": "preferences", "evidence": "turn_7"},
            {"question": "What evening class does the user teach?", "answer": "pottery", "category": "profession", "evidence": "turn_5"},
        ],
    },
]


def load_locomo(limit: int = 30, dry_run: bool = False):
    """
    LoCoMo → PostgreSQL memory_store + Milvus (PARTITIONED)
           + experiments/results/locomo_probes.csv

    Tries HuggingFace snap-research/locomo; falls back to _SYNTHETIC_PROBES
    (5 users × 10 turns + 5 QA pairs each = 50 memories, 25 probes).

    Conversation turns → memory_store + Milvus (FACTUAL, importance=0.70)
    QA pairs → locomo_probes.csv  (consumed by benchmarks.py for Recall@k)

    Why these DBs?
    • PostgreSQL  – tracks access_count; shows which turns are retrieved.
    • Milvus      – semantic search for QA queries, measures Recall@k.

    Analysis unlocked:
    • Recall Accuracy = (QA pairs where top-k contains ground truth) / total
    """
    console.rule("[bold cyan]4 / 4  LoCoMo → memory_store + Milvus + probe CSV[/bold cyan]")

    if dry_run:
        n_users  = min(limit, len(_SYNTHETIC_PROBES))
        pg_est   = n_users * 10
        probe_est = n_users * 5
        console.print("  [DRY RUN] Would load LoCoMo / synthetic probes (no network, no writes)")
        _print_stats("LoCoMo", {"pg_rows": pg_est, "milvus_rows": pg_est, "probe_rows": probe_est, "users": n_users},
                     dry_run=True, extra={"QA probe rows → CSV": probe_est})
        return pg_est

    # ── Try HuggingFace, fall back to synthetic ──────────────────────────────
    hf_records = []
    try:
        from datasets import load_dataset
        console.print("  Attempting HuggingFace snap-research/locomo (streaming)…")
        ds = load_dataset("snap-research/locomo", split="test", streaming=True)
        for row in ds:
            hf_records.append(row)
            if len(hf_records) >= limit:
                break
        console.print(f"  [green]✓ Loaded {len(hf_records)} LoCoMo records from HuggingFace[/green]")
    except Exception as e:
        console.print(f"  [yellow]LoCoMo unavailable ({e.__class__.__name__}), using synthetic probes.[/yellow]")

    use_synthetic = len(hf_records) == 0

    conn   = get_connection()
    stats  = {"pg_rows": 0, "milvus_rows": 0, "probe_rows": 0, "users": 0}
    probes = []

    # ── Process records ──────────────────────────────────────────────────────
    if use_synthetic:
        records_to_process = _SYNTHETIC_PROBES[:min(limit, len(_SYNTHETIC_PROBES))]
    else:
        records_to_process = hf_records

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(), console=console,
    ) as progress:
        task = progress.add_task("Embedding + writing LoCoMo…", total=len(records_to_process))

        for idx, row in enumerate(records_to_process):
            user_id = f"locomo_user_{idx:04d}"
            stats["users"] += 1

            # ── Resolve turn list ────────────────────────────────────────────
            if use_synthetic:
                turn_texts = row["turns"]
                qa_list    = row["qa"]
            else:
                raw_conv   = row.get("conversation", [])
                turn_texts = []
                for turn in raw_conv:
                    if isinstance(turn, dict):
                        speaker = turn.get("speaker", "")
                        text    = turn.get("text", "").strip()
                        turn_texts.append(f"[{speaker}] {text}" if speaker else text)
                    elif isinstance(turn, str):
                        turn_texts.append(turn.strip())
                qa_list = row.get("qa", [])

            # ── Store turns as memories ──────────────────────────────────────
            for content in turn_texts:
                if len(content) < 20:
                    continue
                try:
                    milvus_id, _ = upsert_memory(
                        content=content,
                        user_id=user_id,
                        memory_type="FACTUAL",
                        importance=0.70,
                        index_strategy="PARTITIONED",
                    )
                    _pg_insert_memory(conn, user_id, "FACTUAL", content, 0.70, milvus_id)
                    stats["pg_rows"]    += 1
                    stats["milvus_rows"] += 1
                except Exception as e:
                    console.print(f"[yellow]  ⚠ {user_id}: {e}[/yellow]")

                if stats["milvus_rows"] % _EMBED_BATCH_SIZE == 0:
                    time.sleep(_EMBED_SLEEP_SEC)

            conn.commit()

            # ── Collect QA probes ────────────────────────────────────────────
            for qa in qa_list:
                if isinstance(qa, dict):
                    question = qa.get("question", "")
                    answer   = qa.get("answer",   "")
                    category = qa.get("category", "")
                    evidence = str(qa.get("evidence", ""))
                    if question and answer:
                        probes.append({
                            "user_id":           user_id,
                            "question":          question,
                            "expected_answer":   answer,
                            "evidence_sessions": evidence,
                            "category":          category,
                        })
                        stats["probe_rows"] += 1

            progress.advance(task)

    conn.close()

    # ── Write probe CSV ──────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    probe_csv = RESULTS_DIR / "locomo_probes.csv"
    if probes:
        with open(probe_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=probes[0].keys())
            writer.writeheader()
            writer.writerows(probes)
        console.print(
            f"  [green]✓ Probe CSV written: {probe_csv}  ({len(probes)} QA pairs)[/green]"
        )

    _print_stats("LoCoMo", stats, dry_run, extra={"QA probe rows → CSV": stats["probe_rows"]})
    return stats["pg_rows"]


# ══════════════════════════════════════════════════════════════════════════════
# Summary printer
# ══════════════════════════════════════════════════════════════════════════════

def _print_stats(name: str, stats: dict, dry_run: bool, extra: dict = None):
    action = "[DRY RUN — no writes]" if dry_run else "written"
    table = Table(title=f"{name} — summary ({action})", show_lines=True)
    table.add_column("Destination", style="cyan")
    table.add_column("Rows", justify="right", style="green")

    if not dry_run:
        table.add_row("PostgreSQL  memory_store",  str(stats.get("pg_rows", 0)))
        table.add_row("Milvus vector store",        str(stats.get("milvus_rows", 0)))
        if extra:
            for k, v in extra.items():
                table.add_row(k, str(v))
    else:
        table.add_row("Would write (pg)",      str(stats.get("pg_rows", 0)))
        table.add_row("Would embed (milvus)",  str(stats.get("milvus_rows", 0)))

    console.print(table)


# ══════════════════════════════════════════════════════════════════════════════
# Final summary across all datasets
# ══════════════════════════════════════════════════════════════════════════════

def _final_summary():
    """Query live DB counts and print a summary table."""
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM memory_store")
            total_memories = cur.fetchone()[0]
            cur.execute("SELECT memory_type, COUNT(*) FROM memory_store GROUP BY memory_type")
            by_type = cur.fetchall()
            cur.execute("SELECT COUNT(*) FROM conflict_log")
            total_conflicts = cur.fetchone()[0]
            cur.execute(
                "SELECT user_id, COUNT(*) AS n FROM memory_store "
                "WHERE user_id LIKE 'vol_%' GROUP BY user_id ORDER BY n"
            )
            volume_users = cur.fetchall()
        conn.close()

        console.rule("[bold green]Database State After Load[/bold green]")

        t1 = Table(title="memory_store", show_lines=True)
        t1.add_column("memory_type", style="cyan")
        t1.add_column("count", justify="right")
        for row in by_type:
            t1.add_row(row[0], str(row[1]))
        t1.add_row("[bold]TOTAL[/bold]", str(total_memories))
        console.print(t1)

        if volume_users:
            t2 = Table(title="Volume-test users (Phase 2.2)", show_lines=True)
            t2.add_column("user_id",       style="cyan")
            t2.add_column("memory count",  justify="right")
            for uid, n in volume_users:
                t2.add_row(uid, str(n))
            console.print(t2)

        console.print(f"\n  conflict_log rows : [green]{total_conflicts}[/green]")
        probe_csv = RESULTS_DIR / "locomo_probes.csv"
        if probe_csv.exists():
            with open(probe_csv) as f:
                probe_count = sum(1 for _ in f) - 1
            console.print(f"  locomo_probes.csv : [green]{probe_count} QA pairs[/green]")

    except Exception as e:
        console.print(f"[yellow]Could not query summary: {e}[/yellow]")


# ══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Load research datasets into PostgreSQL + Milvus"
    )
    parser.add_argument(
        "--dataset",
        choices=["all", "personachat", "dailydialog", "msc", "locomo"],
        default="all",
        help="Which dataset to load (default: all)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap number of records per dataset (default: per-dataset sensible defaults)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print counts without writing to any database",
    )
    args = parser.parse_args()

    dry = args.dry_run
    lim = args.limit

    if dry:
        console.print("[bold yellow]DRY RUN — no data will be written.[/bold yellow]\n")

    start = time.perf_counter()

    if args.dataset in ("all", "personachat"):
        load_personachat(limit=lim or 200, dry_run=dry)
    if args.dataset in ("all", "dailydialog"):
        load_dailydialog(limit=lim or 550, dry_run=dry)
    if args.dataset in ("all", "msc"):
        load_msc(limit=lim or 50, dry_run=dry)
    if args.dataset in ("all", "locomo"):
        load_locomo(limit=lim or 30, dry_run=dry)

    elapsed = time.perf_counter() - start
    console.print(f"\n[bold green]✅ Data load complete in {elapsed:.1f}s[/bold green]")

    if not dry:
        _final_summary()


if __name__ == "__main__":
    main()
