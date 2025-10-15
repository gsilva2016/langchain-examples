from datetime import datetime
from google.adk.tools.tool_context import ToolContext
from collections import defaultdict
import csv

def generate_end_of_day_report(tool_context: ToolContext) -> dict:
    """
    Generates an end-of-day CSV report for all agent activities. 
    Args:
        tool_context: ADK-injected context object providing session state.

    Returns:
        dict: A dictionary containing status and output file path.
    """
    output_csv_path = tool_context.state.get("output_csv_path", "default.csv")
    collection_data = tool_context.state.get("collection_data", [])
  
    hourly_agents_count = defaultdict(int)
    hourly_deliveries_count = defaultdict(int)
    hourly_price_alert_count = defaultdict(int)

    overall_price_summaries = []
    overall_idle_summaries = []


    for item in collection_data:
        metadata = item.get('metadata', {})

        # Extract relevant fields
        price_time_str = metadata.get('price_alert_time')
        available_agents = metadata.get('available_agents')
        price_alert = metadata.get('price_alert_status')
        deliveries_count = metadata.get('deliveries_count')
        price_alert_summary = metadata.get('price_alert_summary')
        summary = metadata.get('summary')
        event_creation_ts = metadata.get('event_creation_timestamp', '')

        # Collect overall textual summaries
        if price_alert_summary:
            overall_price_summaries.append(price_alert_summary)
        if summary:
            overall_idle_summaries.append(summary)
        # Parse event_creation_timestamp to hour bucket
        if event_creation_ts:
            try:
                event_time = datetime.strptime(event_creation_ts, '%Y-%m-%dT%H:%M:%S')
                event_hour = event_time.replace(minute=0, second=0, microsecond=0)
                hourly_agents_count[event_hour] = max(hourly_agents_count[event_hour], available_agents)
                hourly_deliveries_count[event_hour] = max(hourly_deliveries_count[event_hour], deliveries_count)

                if price_alert:
                    hourly_price_alert_count[event_hour] += 1
            except Exception:
                continue

    # Sort hours ascending for reporting
    sorted_hours = sorted(hourly_agents_count.keys())

    # Prepare CSV rows
    rows = []
    for hour in sorted_hours:
        agents = hourly_agents_count[hour]
        deliveries = hourly_deliveries_count[hour]
        alerts = hourly_price_alert_count[hour]
        ratio = float(agents) / deliveries if deliveries > 0 else 1.0
        rows.append({
            'Hour': hour.strftime('%Y-%m-%d %H:%M'),
            'Maximum Available Agents': agents,
            'Maximum Deliveries Count': deliveries,
            'Price Alert Count': alerts,
            'Agent to Delivery Ratio': f"{ratio:.2f}",
        })

    # Write to CSV file
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Hour', 'Maximum Available Agents', 'Maximum Deliveries Count', 'Price Alert Count', 'Agent to Delivery Ratio']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(rows)

        # Write overall summaries as additional lines
        writer.writerow({})
        writer.writerow({'Hour': 'Overall Alert Summaries:'})
        for summary in overall_price_summaries:
            writer.writerow({'Hour': summary})
        writer.writerow({})
        writer.writerow({'Hour': 'Overall Idling Summaries:'})
        for summary in overall_idle_summaries:
            writer.writerow({'Hour': summary})

    return {"status": "success", "output_csv_path": output_csv_path}