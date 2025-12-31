"""
Alexandra AI - Travel App
Dedicated travel interface for content creation and knowledge management
"""

import gradio as gr
import os
import sys
import json
from datetime import datetime

# Add paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from travel_config import *
from travel_knowledge import TravelKnowledge, get_travel_knowledge
from travel_training_data import TravelTrainingDataGenerator
from travel_content import TravelContentCreator, quick_destination_intro, quick_outro
from travel_research import TravelResearcher, get_researcher

# Initialize
knowledge = get_travel_knowledge()
data_generator = TravelTrainingDataGenerator()
content_creator = TravelContentCreator()
researcher = get_researcher()

# ============== Knowledge Base Functions ==============

def add_destination(name, country, rating, highlights, tips, budget, would_return):
    """Add a destination to knowledge base"""
    if not name or not country:
        return "Please provide destination name and country"

    data = {
        "rating": int(rating) if rating else 7,
        "highlights": [h.strip() for h in highlights.split(",") if h.strip()] if highlights else [],
        "tips": tips,
        "budget_per_day": float(budget) if budget else 100,
        "would_return": would_return,
        "visited_date": datetime.now().strftime("%Y-%m"),
    }

    result = knowledge.add_destination(name, country, data)
    return result + f"\n\nTotal destinations: {knowledge.stats()['destinations']}"

def add_experience(destination, title, story, category):
    """Add a travel experience"""
    if not destination or not title or not story:
        return "Please fill in all fields"

    result = knowledge.add_experience(destination, title, story, category)
    return result

def add_tip(category, tip, destination):
    """Add a travel tip"""
    if not category or not tip:
        return "Please provide category and tip"

    result = knowledge.add_tip(category, tip, destination if destination else None)
    return result

def add_favorite(category, name, location, notes):
    """Add a favorite place"""
    if not category or not name or not location:
        return "Please fill in required fields"

    result = knowledge.add_favorite(category, name, location, notes)
    return result

def get_stats():
    """Get knowledge base stats"""
    stats = knowledge.stats()
    return f"""**Travel Knowledge Base Stats**

- Destinations visited: {stats['destinations']}
- Countries: {stats['countries']}
- Experiences recorded: {stats['experiences']}
- Tips saved: {stats['tips']}
- Favorite places: {stats['favorites']}
"""

def search_destinations(query, min_rating, max_budget):
    """Search destinations"""
    criteria = {}
    if min_rating:
        criteria['min_rating'] = int(min_rating)
    if max_budget:
        criteria['max_budget'] = float(max_budget)

    results = knowledge.search_destinations(criteria)

    if not results:
        return "No destinations found matching criteria"

    output = f"**Found {len(results)} destinations:**\n\n"
    for dest in results:
        output += f"**{dest['name']}, {dest['country']}** - {dest.get('rating', 'N/A')}/10\n"
        if dest.get('highlights'):
            output += f"  Highlights: {', '.join(dest['highlights'][:3])}\n"
        output += "\n"

    return output

# ============== Training Data Functions ==============

def generate_training_data():
    """Generate training data from knowledge base"""
    data = data_generator.generate_from_knowledge(knowledge)
    if data:
        filepath = data_generator.save_training_data(data, "from_knowledge.jsonl")
        return f"Generated {len(data)} training examples\nSaved to: {filepath}"
    return "No data generated - add more to your knowledge base first!"

def generate_templates(num_examples):
    """Generate template examples to fill in"""
    templates = data_generator.generate_synthetic_data(int(num_examples))
    filepath = data_generator.save_training_data(templates, "templates_to_fill.jsonl")
    return f"Generated {len(templates)} templates\nSaved to: {filepath}\n\nEdit these files to add your real knowledge!"

def export_for_training(format_type):
    """Export training data in specified format"""
    # Load existing data
    data = data_generator.load_training_data("from_knowledge.jsonl")
    if not data:
        return "No training data found. Generate from knowledge base first!"

    filepath = data_generator.export_for_finetuning(data, format_type)
    return f"Exported {len(data)} examples in {format_type} format\nSaved to: {filepath}"

# ============== Content Creation Functions ==============

def create_review_script(destination):
    """Create a destination review script"""
    if not destination:
        return "Please enter a destination name", ""

    script = content_creator.generate_script("destination_review", destination=destination)
    filepath = content_creator.save_script(script)

    # Generate intro
    intro = quick_destination_intro(destination, "")

    output = f"**Script created for: {destination}**\n\n"
    output += f"Saved to: {filepath}\n\n"
    output += f"**Sections ({len(script['sections'])}):**\n"
    for s in script['sections']:
        output += f"- {s['name']} ({s['duration_hint']})\n"

    return output, intro

def create_tips_script(topic, num_tips):
    """Create a tips video script"""
    if not topic:
        return "Please enter a topic", ""

    script = content_creator.generate_script(
        "travel_tips",
        topic=topic,
        count=int(num_tips)
    )
    filepath = content_creator.save_script(script)

    output = f"**Tips script created for: {topic}**\n\n"
    output += f"Saved to: {filepath}\n"
    output += f"Sections: {len(script['sections'])}"

    return output, ""

def create_comparison_script(dest1, dest2):
    """Create a comparison video script"""
    if not dest1 or not dest2:
        return "Please enter both destinations", ""

    script = content_creator.generate_script(
        "comparison",
        destination1=dest1,
        destination2=dest2
    )
    filepath = content_creator.save_script(script)

    output = f"**Comparison script: {dest1} vs {dest2}**\n\n"
    output += f"Saved to: {filepath}\n"
    output += f"Sections: {len(script['sections'])}"

    return output, ""

def create_itinerary_script(destination, duration):
    """Create an itinerary script"""
    if not destination or not duration:
        return "Please fill in all fields", ""

    script = content_creator.generate_script(
        "itinerary",
        destination=destination,
        duration=duration
    )
    filepath = content_creator.save_script(script)

    output = f"**Itinerary script: {duration} in {destination}**\n\n"
    output += f"Saved to: {filepath}\n"
    output += f"Sections: {len(script['sections'])}"

    return output, ""

def list_scripts():
    """List all saved scripts"""
    scripts = []
    for f in os.listdir(SCRIPTS_DIR):
        if f.endswith('.json'):
            scripts.append(f)

    if not scripts:
        return "No scripts saved yet"

    return "**Saved Scripts:**\n\n" + "\n".join(f"- {s}" for s in scripts)

# ============== Research Functions ==============

def research_destination(destination, research_type):
    """Research a destination"""
    if not destination:
        return "Please enter a destination"

    result = researcher.search_destination(destination, research_type)
    results = result.get("results", [])

    if not results:
        return f"No results found for {destination} ({research_type})"

    output = f"**{destination} - {research_type.title()}**\n\n"
    for r in results:
        output += f"**{r.get('title', 'No title')}**\n"
        output += f"{r.get('snippet', r.get('body', 'No description'))}\n"
        if r.get('url'):
            output += f"[Link]({r.get('url')})\n"
        output += "\n---\n"

    return output

def full_research(destination):
    """Do comprehensive research on a destination"""
    if not destination:
        return "Please enter a destination"

    research = researcher.research_destination(destination)
    sections = research.get("sections", {})

    output = f"# Complete Research: {destination}\n\n"

    for section_name, results in sections.items():
        output += f"## {section_name.title()}\n\n"
        for r in results[:3]:  # Top 3 per section
            output += f"- **{r.get('title', 'No title')}**: {r.get('snippet', '')[:150]}...\n"
        output += "\n"

    return output

def create_trip_plan(name, destinations, dates, notes):
    """Create a new trip plan"""
    if not name or not destinations:
        return "Please provide trip name and destinations", ""

    dest_list = [d.strip() for d in destinations.split(",") if d.strip()]
    trip_id = researcher.create_trip(name, dest_list, dates, notes)

    return f"Created trip: **{name}**\nID: {trip_id}\nDestinations: {', '.join(dest_list)}", trip_id

def list_trips():
    """List all trip plans"""
    trips = researcher.list_trips()

    if not trips:
        return "No trips planned yet"

    output = "**Your Trip Plans:**\n\n"
    for trip in trips:
        output += f"- **{trip['name']}** ({trip['id']})\n"
        output += f"  Destinations: {', '.join(trip['destinations'])}\n"
        if trip.get('dates'):
            output += f"  Dates: {trip['dates']}\n"
        output += "\n"

    return output

def get_trip_details(trip_id):
    """Get details of a trip"""
    trip = researcher.get_trip(trip_id)

    if not trip:
        return "Trip not found"

    output = f"# {trip['name']}\n\n"
    output += f"**Destinations:** {', '.join(trip['destinations'])}\n"
    output += f"**Dates:** {trip.get('dates', 'Not set')}\n"
    output += f"**Notes:** {trip.get('notes', 'None')}\n\n"

    # Itinerary
    if trip.get("itinerary"):
        output += "## Itinerary\n\n"
        for day in trip["itinerary"]:
            output += f"**Day {day['day']}:**\n"
            for activity in day.get("activities", []):
                output += f"  - {activity}\n"
            output += "\n"

    # Bookings
    if trip.get("bookings"):
        output += "## Bookings\n\n"
        for booking in trip["bookings"]:
            output += f"- **{booking['type']}:** {booking['name']}"
            if booking.get('confirmation'):
                output += f" (#{booking['confirmation']})"
            if booking.get('cost'):
                output += f" - ${booking['cost']}"
            output += "\n"

    # Budget
    if trip.get("budget", {}).get("breakdown"):
        output += "\n## Budget\n\n"
        for cat, amount in trip["budget"]["breakdown"].items():
            output += f"- {cat}: ${amount}\n"
        output += f"\n**Total Estimated:** ${trip['budget'].get('estimated', 0)}\n"

    # Packing
    if trip.get("packing_list"):
        output += "\n## Packing List\n\n"
        for item in trip["packing_list"]:
            check = "✓" if item.get("packed") else "○"
            output += f"{check} {item['item']} ({item.get('category', 'general')})\n"

    return output

def add_itinerary_item(trip_id, day, activities):
    """Add to trip itinerary"""
    if not trip_id or not day or not activities:
        return "Please fill in all fields"

    activity_list = [a.strip() for a in activities.split(",") if a.strip()]
    success = researcher.add_to_itinerary(trip_id, int(day), activity_list)

    if success:
        return f"Added {len(activity_list)} activities to Day {day}"
    return "Failed to add activities"

def add_booking_to_trip(trip_id, booking_type, name, confirmation, dates, cost):
    """Add booking to trip"""
    if not trip_id or not booking_type or not name:
        return "Please fill in required fields"

    success = researcher.add_booking(
        trip_id, booking_type, name,
        confirmation or "", dates or "",
        float(cost) if cost else 0
    )

    if success:
        return f"Added {booking_type}: {name}"
    return "Failed to add booking"

def suggest_destinations_search(budget, trip_type, duration):
    """Get destination suggestions"""
    preferences = {
        "budget": budget,
        "type": trip_type,
        "duration": duration
    }

    results = researcher.suggest_destinations(preferences)

    if not results:
        return "No suggestions found (web search may not be available)"

    output = f"**Suggested Destinations for {trip_type} ({budget}):**\n\n"
    for r in results:
        output += f"- **{r.get('title', 'No title')}**\n"
        output += f"  {r.get('snippet', '')[:200]}...\n\n"

    return output

# ============== Gradio UI ==============

with gr.Blocks(title="Alexandra AI - Travel", theme=gr.themes.Soft()) as app:

    gr.Markdown("# Alexandra AI - Travel Module")
    gr.Markdown("*Build your travel knowledge, create content, and prepare training data*")

    with gr.Tabs():

        # ============== Research Tab ==============
        with gr.Tab("Research & Plan"):
            gr.Markdown("### Research Destinations & Plan Trips")

            with gr.Accordion("Research a Destination", open=True):
                with gr.Row():
                    research_dest = gr.Textbox(label="Destination", placeholder="Tokyo, Japan")
                    research_type = gr.Dropdown(
                        choices=["general", "food", "hotels", "activities", "budget",
                                "safety", "transportation", "hidden_gems", "nightlife"],
                        value="general",
                        label="What to Research"
                    )
                with gr.Row():
                    research_btn = gr.Button("Search", variant="primary")
                    full_research_btn = gr.Button("Full Research (all categories)")
                research_results = gr.Markdown()

                research_btn.click(
                    research_destination,
                    inputs=[research_dest, research_type],
                    outputs=[research_results]
                )
                full_research_btn.click(
                    full_research,
                    inputs=[research_dest],
                    outputs=[research_results]
                )

            with gr.Accordion("Get Destination Suggestions", open=False):
                with gr.Row():
                    suggest_budget = gr.Dropdown(
                        choices=["budget", "mid-range", "luxury"],
                        value="mid-range",
                        label="Budget"
                    )
                    suggest_type = gr.Dropdown(
                        choices=["beach", "city", "nature", "culture", "adventure", "romantic"],
                        value="beach",
                        label="Trip Type"
                    )
                    suggest_duration = gr.Dropdown(
                        choices=["weekend", "week", "two weeks", "month"],
                        value="week",
                        label="Duration"
                    )
                suggest_btn = gr.Button("Get Suggestions")
                suggest_results = gr.Markdown()
                suggest_btn.click(
                    suggest_destinations_search,
                    inputs=[suggest_budget, suggest_type, suggest_duration],
                    outputs=[suggest_results]
                )

            gr.Markdown("---")
            gr.Markdown("### Trip Planning")

            with gr.Accordion("Create New Trip", open=False):
                trip_name = gr.Textbox(label="Trip Name", placeholder="Summer in Europe")
                trip_destinations = gr.Textbox(label="Destinations (comma-separated)", placeholder="Paris, Barcelona, Rome")
                trip_dates = gr.Textbox(label="Dates", placeholder="June 15-30, 2025")
                trip_notes = gr.Textbox(label="Notes", lines=2)
                create_trip_btn = gr.Button("Create Trip", variant="primary")
                create_trip_result = gr.Markdown()
                current_trip_id = gr.Textbox(label="Current Trip ID", visible=True)

                create_trip_btn.click(
                    create_trip_plan,
                    inputs=[trip_name, trip_destinations, trip_dates, trip_notes],
                    outputs=[create_trip_result, current_trip_id]
                )

            with gr.Accordion("Manage Trip", open=False):
                manage_trip_id = gr.Textbox(label="Trip ID", placeholder="trip_1")
                view_trip_btn = gr.Button("View Trip Details")
                trip_details = gr.Markdown()
                view_trip_btn.click(get_trip_details, inputs=[manage_trip_id], outputs=[trip_details])

                gr.Markdown("**Add to Itinerary**")
                with gr.Row():
                    itin_day = gr.Number(label="Day", value=1, precision=0)
                    itin_activities = gr.Textbox(label="Activities (comma-separated)")
                add_itin_btn = gr.Button("Add to Day")
                itin_result = gr.Textbox(label="Result")
                add_itin_btn.click(
                    add_itinerary_item,
                    inputs=[manage_trip_id, itin_day, itin_activities],
                    outputs=[itin_result]
                )

                gr.Markdown("**Add Booking**")
                with gr.Row():
                    booking_type = gr.Dropdown(
                        choices=["flight", "hotel", "activity", "restaurant", "transport", "other"],
                        label="Type"
                    )
                    booking_name = gr.Textbox(label="Name")
                with gr.Row():
                    booking_confirm = gr.Textbox(label="Confirmation #")
                    booking_dates = gr.Textbox(label="Dates")
                    booking_cost = gr.Number(label="Cost ($)")
                add_booking_btn = gr.Button("Add Booking")
                booking_result = gr.Textbox(label="Result")
                add_booking_btn.click(
                    add_booking_to_trip,
                    inputs=[manage_trip_id, booking_type, booking_name, booking_confirm, booking_dates, booking_cost],
                    outputs=[booking_result]
                )

            with gr.Accordion("Your Trips", open=False):
                trips_list = gr.Markdown()
                refresh_trips_btn = gr.Button("Refresh Trip List")
                refresh_trips_btn.click(list_trips, outputs=[trips_list])

        # ============== Knowledge Tab ==============
        with gr.Tab("My Experiences"):
            gr.Markdown("### Add Your Travel Experiences")

            with gr.Accordion("Add Destination", open=True):
                with gr.Row():
                    dest_name = gr.Textbox(label="Destination Name", placeholder="Barcelona")
                    dest_country = gr.Textbox(label="Country", placeholder="Spain")

                with gr.Row():
                    dest_rating = gr.Slider(1, 10, value=7, step=1, label="Rating")
                    dest_budget = gr.Number(label="Budget/Day ($)", value=100)
                    dest_return = gr.Checkbox(label="Would Return?", value=True)

                dest_highlights = gr.Textbox(
                    label="Highlights (comma-separated)",
                    placeholder="Beach, Food, Architecture"
                )
                dest_tips = gr.Textbox(label="Your Tips", lines=2)
                add_dest_btn = gr.Button("Add Destination", variant="primary")
                dest_result = gr.Textbox(label="Result")

                add_dest_btn.click(
                    add_destination,
                    inputs=[dest_name, dest_country, dest_rating, dest_highlights,
                           dest_tips, dest_budget, dest_return],
                    outputs=[dest_result]
                )

            with gr.Accordion("Add Experience/Story", open=False):
                exp_dest = gr.Textbox(label="Destination")
                exp_title = gr.Textbox(label="Title", placeholder="Getting Lost in the Old Town")
                exp_story = gr.Textbox(label="Your Story", lines=4)
                exp_category = gr.Dropdown(
                    choices=["adventure", "food", "culture", "nightlife", "nature", "general"],
                    value="general",
                    label="Category"
                )
                add_exp_btn = gr.Button("Add Experience")
                exp_result = gr.Textbox(label="Result")

                add_exp_btn.click(
                    add_experience,
                    inputs=[exp_dest, exp_title, exp_story, exp_category],
                    outputs=[exp_result]
                )

            with gr.Accordion("Add Tips & Favorites", open=False):
                with gr.Row():
                    with gr.Column():
                        tip_category = gr.Dropdown(
                            choices=["packing", "budgeting", "safety", "food", "transportation", "general"],
                            label="Tip Category"
                        )
                        tip_text = gr.Textbox(label="Tip")
                        tip_dest = gr.Textbox(label="Destination (optional)")
                        add_tip_btn = gr.Button("Add Tip")

                    with gr.Column():
                        fav_category = gr.Dropdown(
                            choices=["restaurants", "hotels", "activities", "bars", "cafes", "shops"],
                            label="Favorite Category"
                        )
                        fav_name = gr.Textbox(label="Name")
                        fav_location = gr.Textbox(label="Location")
                        fav_notes = gr.Textbox(label="Notes")
                        add_fav_btn = gr.Button("Add Favorite")

                tip_fav_result = gr.Textbox(label="Result")
                add_tip_btn.click(add_tip, inputs=[tip_category, tip_text, tip_dest], outputs=[tip_fav_result])
                add_fav_btn.click(add_favorite, inputs=[fav_category, fav_name, fav_location, fav_notes], outputs=[tip_fav_result])

            with gr.Accordion("Search & Stats", open=False):
                stats_display = gr.Markdown()
                refresh_stats = gr.Button("Refresh Stats")
                refresh_stats.click(get_stats, outputs=[stats_display])

                gr.Markdown("---")
                with gr.Row():
                    search_min_rating = gr.Slider(1, 10, value=1, label="Min Rating")
                    search_max_budget = gr.Number(label="Max Budget/Day", value=500)
                search_btn = gr.Button("Search Destinations")
                search_results = gr.Markdown()
                search_btn.click(
                    search_destinations,
                    inputs=[gr.Textbox(visible=False), search_min_rating, search_max_budget],
                    outputs=[search_results]
                )

        # ============== Content Creation Tab ==============
        with gr.Tab("Content Creation"):
            gr.Markdown("### Create Travel Video Scripts")

            with gr.Accordion("Destination Review", open=True):
                review_dest = gr.Textbox(label="Destination", placeholder="Barcelona")
                review_btn = gr.Button("Create Review Script", variant="primary")
                review_result = gr.Markdown()
                review_intro = gr.Textbox(label="Sample Intro", lines=4)
                review_btn.click(create_review_script, inputs=[review_dest], outputs=[review_result, review_intro])

            with gr.Accordion("Tips Video", open=False):
                tips_topic = gr.Textbox(label="Topic", placeholder="Packing Light")
                tips_count = gr.Slider(3, 15, value=10, step=1, label="Number of Tips")
                tips_btn = gr.Button("Create Tips Script")
                tips_result = gr.Markdown()
                tips_preview = gr.Textbox(label="Preview", lines=4)
                tips_btn.click(create_tips_script, inputs=[tips_topic, tips_count], outputs=[tips_result, tips_preview])

            with gr.Accordion("Destination Comparison", open=False):
                with gr.Row():
                    comp_dest1 = gr.Textbox(label="Destination 1", placeholder="Barcelona")
                    comp_dest2 = gr.Textbox(label="Destination 2", placeholder="Madrid")
                comp_btn = gr.Button("Create Comparison Script")
                comp_result = gr.Markdown()
                comp_preview = gr.Textbox(label="Preview", lines=4)
                comp_btn.click(create_comparison_script, inputs=[comp_dest1, comp_dest2], outputs=[comp_result, comp_preview])

            with gr.Accordion("Itinerary", open=False):
                itin_dest = gr.Textbox(label="Destination", placeholder="Tokyo")
                itin_duration = gr.Textbox(label="Duration", placeholder="5 days")
                itin_btn = gr.Button("Create Itinerary Script")
                itin_result = gr.Markdown()
                itin_preview = gr.Textbox(label="Preview", lines=4)
                itin_btn.click(create_itinerary_script, inputs=[itin_dest, itin_duration], outputs=[itin_result, itin_preview])

            gr.Markdown("---")
            scripts_list = gr.Markdown()
            list_scripts_btn = gr.Button("List Saved Scripts")
            list_scripts_btn.click(list_scripts, outputs=[scripts_list])

        # ============== Training Data Tab ==============
        with gr.Tab("Training Data"):
            gr.Markdown("### Generate Fine-Tuning Data")
            gr.Markdown("Build training data from your knowledge base for fine-tuning your LLM on travel.")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("**From Your Knowledge**")
                    gr.Markdown("Generate Q&A pairs from your saved destinations, experiences, and tips.")
                    gen_from_knowledge_btn = gr.Button("Generate from Knowledge", variant="primary")
                    gen_knowledge_result = gr.Textbox(label="Result", lines=3)
                    gen_from_knowledge_btn.click(generate_training_data, outputs=[gen_knowledge_result])

                with gr.Column():
                    gr.Markdown("**Template Generator**")
                    gr.Markdown("Create template questions to fill in with your answers.")
                    template_count = gr.Slider(10, 200, value=50, step=10, label="Number of Templates")
                    gen_templates_btn = gr.Button("Generate Templates")
                    gen_templates_result = gr.Textbox(label="Result", lines=3)
                    gen_templates_btn.click(generate_templates, inputs=[template_count], outputs=[gen_templates_result])

            gr.Markdown("---")
            gr.Markdown("### Export for Training")
            export_format = gr.Radio(choices=["openai", "llama"], value="llama", label="Format")
            export_btn = gr.Button("Export Training Data")
            export_result = gr.Textbox(label="Result", lines=3)
            export_btn.click(export_for_training, inputs=[export_format], outputs=[export_result])

            gr.Markdown(f"""
---
### Training Data Location
All training data is saved to: `{TRAINING_DATA_DIR}`

After generating data:
1. Review and edit the generated files
2. Add your personal answers to template questions
3. Export in your preferred format
4. Use for fine-tuning after your current training completes
            """)

        # ============== Help Tab ==============
        with gr.Tab("Help"):
            gr.Markdown("""
# Travel Module Help

## Knowledge Base
Add your personal travel experiences to build Alexandra's travel expertise:
- **Destinations**: Places you've visited with ratings and tips
- **Experiences**: Stories and memorable moments
- **Tips**: Practical advice organized by category
- **Favorites**: Best restaurants, hotels, activities

## Content Creation
Generate video script templates:
- **Review**: Full destination guide with all sections
- **Tips**: Numbered list format for quick tips videos
- **Comparison**: Side-by-side destination comparison
- **Itinerary**: Day-by-day travel plan

Scripts are saved as JSON files that you edit, then use with batch video generation.

## Training Data
Prepare data for fine-tuning:
1. **From Knowledge**: Converts your saved experiences into Q&A pairs
2. **Templates**: Generates question templates for you to fill in
3. **Export**: Formats data for Llama or OpenAI fine-tuning

## Workflow

1. **Add your travel knowledge** (Knowledge Base tab)
2. **Generate training data** (Training Data tab)
3. **Review and enhance** the generated data
4. **Fine-tune** your model after current training completes
5. **Create content** using scripts (Content Creation tab)
6. **Generate videos** using the main Alexandra app
            """)

    # Load stats on startup
    app.load(get_stats, outputs=[stats_display])

if __name__ == "__main__":
    print("="*50)
    print("Alexandra AI - Travel Module")
    print("="*50)
    print(f"Knowledge: {knowledge.stats()}")
    print(f"Scripts dir: {SCRIPTS_DIR}")
    print(f"Training data dir: {TRAINING_DATA_DIR}")
    print("="*50)

    app.launch(server_name="0.0.0.0", server_port=7863)
