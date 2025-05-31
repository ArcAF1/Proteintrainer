"""
Research Logger for Autonomous Research System
Provides detailed hourly logging and progress tracking
"""
from __future__ import annotations

import json
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ResearchLogger:
    """
    Manages research logs with hourly updates and detailed progress tracking.
    Creates human-readable logs that show the AI's research process.
    """
    
    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Log files
        self.hourly_log_file = None
        self.event_log_file = None
        self.current_project = None
        
    def _get_project_log_dir(self, project_name: str) -> Path:
        """Get the log directory for a specific project."""
        project_dir = self.log_dir / project_name
        project_dir.mkdir(exist_ok=True)
        return project_dir
        
    def _get_hourly_log_path(self, project_name: str) -> Path:
        """Get the hourly log file path."""
        return self._get_project_log_dir(project_name) / "hourly_log.md"
        
    def _get_event_log_path(self, project_name: str) -> Path:
        """Get the event log file path."""
        return self._get_project_log_dir(project_name) / "event_log.json"
        
    async def log_event(self, project_name: str, event_type: str, data: Dict[str, Any]):
        """Log a research event."""
        event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'data': data
        }
        
        # Append to event log
        event_log_path = self._get_event_log_path(project_name)
        
        # Read existing events
        events = []
        if event_log_path.exists():
            with open(event_log_path, 'r') as f:
                events = json.load(f)
                
        # Append new event
        events.append(event)
        
        # Write back
        with open(event_log_path, 'w') as f:
            json.dump(events, f, indent=2)
            
        logger.info(f"Logged event: {event_type} for project {project_name}")
        
    async def log_hourly_update(self, project_name: str, update: Dict[str, Any]):
        """Log hourly progress update in human-readable format."""
        hourly_log_path = self._get_hourly_log_path(project_name)
        
        # Format the update as markdown
        markdown_update = self._format_hourly_update(update)
        
        # Append to hourly log
        with open(hourly_log_path, 'a') as f:
            f.write(markdown_update)
            f.write("\n\n---\n\n")
            
        # Also save as JSON for programmatic access
        json_path = self._get_project_log_dir(project_name) / "hourly_updates.json"
        updates = []
        if json_path.exists():
            with open(json_path, 'r') as f:
                updates = json.load(f)
        updates.append(update)
        with open(json_path, 'w') as f:
            json.dump(updates, f, indent=2)
            
    def _format_hourly_update(self, update: Dict[str, Any]) -> str:
        """Format hourly update as readable markdown."""
        timestamp = update.get('timestamp', datetime.now().isoformat())
        
        markdown = f"## 📊 Hourly Update - {timestamp}\n\n"
        
        # Project and phase info
        markdown += f"**Project:** {update.get('project', 'Unknown')}\n"
        markdown += f"**Current Phase:** {update.get('current_phase', 'Unknown')}\n"
        markdown += f"**Iteration:** {update.get('iteration', 0)}\n\n"
        
        # Recent actions
        markdown += "### 🔄 Recent Actions\n"
        markdown += f"{update.get('recent_actions', 'No recent actions logged')}\n\n"
        
        # Key findings
        markdown += "### 🔍 Key Findings\n"
        findings = update.get('key_findings', [])
        if findings:
            for finding in findings:
                markdown += f"- {finding}\n"
        else:
            markdown += "- No significant findings this hour\n"
        markdown += "\n"
        
        # Current hypothesis
        markdown += "### 💡 Current Hypothesis\n"
        hypothesis = update.get('current_hypothesis', 'No hypothesis currently being tested')
        markdown += f"{hypothesis}\n\n"
        
        # Thought process
        markdown += "### 🧠 Thought Process\n"
        markdown += f"{update.get('thought_process', 'Processing...')}\n\n"
        
        # Next steps
        markdown += "### ➡️ Next Steps\n"
        markdown += f"{update.get('next_steps', 'Planning next phase...')}\n"
        
        return markdown
        
    async def log_phase_start(self, project_name: str, phase: str, data: Dict[str, Any]):
        """Log the start of a research phase."""
        await self.log_event(project_name, "PHASE_START", {
            'phase': phase,
            **data
        })
        
        # Also add to hourly log
        hourly_path = self._get_hourly_log_path(project_name)
        with open(hourly_path, 'a') as f:
            f.write(f"\n### 🚀 Started Phase: {phase}\n")
            f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            if data:
                f.write("Parameters:\n")
                for key, value in data.items():
                    f.write(f"- {key}: {value}\n")
            f.write("\n")
            
    async def log_phase_complete(self, project_name: str, phase: str, results: Dict[str, Any]):
        """Log the completion of a research phase."""
        await self.log_event(project_name, "PHASE_COMPLETE", {
            'phase': phase,
            **results
        })
        
        # Add to hourly log
        hourly_path = self._get_hourly_log_path(project_name)
        with open(hourly_path, 'a') as f:
            f.write(f"\n### ✅ Completed Phase: {phase}\n")
            f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            if results:
                f.write("Results:\n")
                for key, value in results.items():
                    f.write(f"- {key}: {value}\n")
            f.write("\n")
            
    async def log_error(self, project_name: str, error: str, phase: str = None):
        """Log an error during research."""
        await self.log_event(project_name, "ERROR", {
            'error': error,
            'phase': phase,
            'timestamp': datetime.now().isoformat()
        })
        
        # Add to hourly log
        hourly_path = self._get_hourly_log_path(project_name)
        with open(hourly_path, 'a') as f:
            f.write(f"\n### ❌ Error Occurred\n")
            f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Phase: {phase or 'Unknown'}\n")
            f.write(f"Error: {error}\n\n")
            
    async def get_recent_logs(self, project_name: str, hours: int = 1) -> List[Dict[str, Any]]:
        """Get recent log entries."""
        event_log_path = self._get_event_log_path(project_name)
        
        if not event_log_path.exists():
            return []
            
        with open(event_log_path, 'r') as f:
            events = json.load(f)
            
        # Filter by time
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_events = []
        
        for event in events:
            event_time = datetime.fromisoformat(event['timestamp'])
            if event_time > cutoff_time:
                recent_events.append(event)
                
        return recent_events
        
    async def create_research_summary(self, project_name: str) -> str:
        """Create a summary of the research project."""
        summary_path = self._get_project_log_dir(project_name) / "research_summary.md"
        
        # Get all events
        event_log_path = self._get_event_log_path(project_name)
        if not event_log_path.exists():
            return "No research data available"
            
        with open(event_log_path, 'r') as f:
            events = json.load(f)
            
        # Create summary
        summary = f"# Research Summary: {project_name}\n\n"
        summary += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Project timeline
        if events:
            start_time = datetime.fromisoformat(events[0]['timestamp'])
            end_time = datetime.fromisoformat(events[-1]['timestamp'])
            duration = end_time - start_time
            
            summary += "## Timeline\n"
            summary += f"- Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            summary += f"- Last Update: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            summary += f"- Duration: {duration}\n\n"
            
        # Phase summary
        summary += "## Research Phases\n"
        phase_counts = {}
        for event in events:
            if event['event_type'] == 'PHASE_COMPLETE':
                phase = event['data'].get('phase', 'Unknown')
                phase_counts[phase] = phase_counts.get(phase, 0) + 1
                
        for phase, count in phase_counts.items():
            summary += f"- {phase}: {count} iterations\n"
        summary += "\n"
        
        # Key milestones
        summary += "## Key Milestones\n"
        for event in events:
            if event['event_type'] in ['PROJECT_START', 'PROJECT_COMPLETE', 'ERROR']:
                summary += f"- {event['timestamp']}: {event['event_type']}\n"
                
        # Save summary
        with open(summary_path, 'w') as f:
            f.write(summary)
            
        return summary
        
    def get_latest_log_content(self, project_name: str, lines: int = 50) -> str:
        """Get the latest content from the hourly log."""
        hourly_log_path = self._get_hourly_log_path(project_name)
        
        if not hourly_log_path.exists():
            return "No hourly log available yet."
            
        with open(hourly_log_path, 'r') as f:
            content = f.read()
            
        # Return last N lines or full content if shorter
        lines_list = content.split('\n')
        if len(lines_list) > lines:
            return '\n'.join(lines_list[-lines:])
        return content
        
    def list_project_logs(self) -> List[str]:
        """List all projects with logs."""
        projects = []
        for path in self.log_dir.iterdir():
            if path.is_dir():
                projects.append(path.name)
        return sorted(projects)
        
    async def export_logs(self, project_name: str, format: str = 'markdown') -> Path:
        """Export all logs for a project."""
        export_dir = self._get_project_log_dir(project_name) / "exports"
        export_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format == 'markdown':
            export_path = export_dir / f"research_log_{timestamp}.md"
            
            # Combine all logs into one markdown file
            content = f"# Complete Research Log: {project_name}\n\n"
            content += f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            
            # Add hourly log
            hourly_log_path = self._get_hourly_log_path(project_name)
            if hourly_log_path.exists():
                with open(hourly_log_path, 'r') as f:
                    content += "## Hourly Progress Log\n\n"
                    content += f.read()
                    
            # Add summary
            summary = await self.create_research_summary(project_name)
            content += "\n\n## Summary\n\n"
            content += summary
            
            with open(export_path, 'w') as f:
                f.write(content)
                
        elif format == 'json':
            export_path = export_dir / f"research_log_{timestamp}.json"
            
            # Export all JSON data
            export_data = {
                'project': project_name,
                'export_time': datetime.now().isoformat(),
                'events': [],
                'hourly_updates': []
            }
            
            # Add events
            event_log_path = self._get_event_log_path(project_name)
            if event_log_path.exists():
                with open(event_log_path, 'r') as f:
                    export_data['events'] = json.load(f)
                    
            # Add hourly updates
            json_path = self._get_project_log_dir(project_name) / "hourly_updates.json"
            if json_path.exists():
                with open(json_path, 'r') as f:
                    export_data['hourly_updates'] = json.load(f)
                    
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2)
                
        return export_path 