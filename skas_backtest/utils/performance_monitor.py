"""
Performance monitoring utilities for tracking API call performance.

This module provides tools to monitor and analyze the performance of NSE API calls
during backtesting to identify bottlenecks and optimization opportunities.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from contextlib import contextmanager
import logging


@dataclass
class APICallStats:
    """Statistics for API calls."""
    count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    slowest_calls: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_call(self, duration: float, cached: bool = False, context: str = ""):
        """Record a single API call."""
        self.count += 1
        self.total_time += duration
        self.min_time = min(self.min_time, duration)
        self.max_time = max(self.max_time, duration)
        
        if cached:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
        
        # Track slowest calls (keep top 5)
        call_info = {
            'duration': duration,
            'context': context,
            'timestamp': time.time()
        }
        
        self.slowest_calls.append(call_info)
        self.slowest_calls.sort(key=lambda x: x['duration'], reverse=True)
        self.slowest_calls = self.slowest_calls[:5]  # Keep only top 5
    
    @property
    def avg_time(self) -> float:
        """Average time per call."""
        return self.total_time / max(self.count, 1)
    
    @property
    def cache_hit_rate(self) -> float:
        """Cache hit rate as percentage."""
        total_requests = self.cache_hits + self.cache_misses
        return (self.cache_hits / max(total_requests, 1)) * 100


class PerformanceMonitor:
    """
    Monitor and track API call performance.
    
    This class provides comprehensive performance monitoring for NSE API calls,
    tracking timing, cache performance, and identifying bottlenecks.
    """
    
    def __init__(self, enable_detailed_logging: bool = False):
        """
        Initialize the performance monitor.
        
        Args:
            enable_detailed_logging: Enable detailed logging of each API call
        """
        self.api_stats: Dict[str, APICallStats] = {}
        self.logger = logging.getLogger(__name__)
        self.enable_detailed_logging = enable_detailed_logging
        self.start_time = time.time()
    
    @contextmanager
    def track_api_call(self, api_name: str, cached: bool = False, context: str = ""):
        """
        Context manager to track API call timing.
        
        Args:
            api_name: Name of the API being called
            cached: Whether this call hit the cache
            context: Additional context (e.g., symbol, date)
        
        Usage:
            with monitor.track_api_call('get_stock_price', context='RELIANCE 2024-01-01'):
                price = get_stock_price('RELIANCE', date(2024, 1, 1))
        """
        start_time = time.perf_counter()
        
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start_time
            self._record_call(api_name, elapsed, cached, context)
    
    def record_cache_hit(self, api_name: str):
        """Record a cache hit for an API call."""
        if api_name not in self.api_stats:
            self.api_stats[api_name] = APICallStats()
        self.api_stats[api_name].cache_hits += 1
    
    def _record_call(self, api_name: str, duration: float, cached: bool, context: str):
        """Record a single API call."""
        if api_name not in self.api_stats:
            self.api_stats[api_name] = APICallStats()
        
        self.api_stats[api_name].add_call(duration, cached, context)
        
        # Log slow calls
        if self.enable_detailed_logging and duration > 0.1:  # > 100ms
            self.logger.warning(f"Slow API call: {api_name} took {duration*1000:.1f}ms [{context}]")
    
    def get_summary(self) -> str:
        """
        Get comprehensive performance summary.
        
        Returns:
            Formatted string with performance statistics
        """
        if not self.api_stats:
            return "No API calls recorded"
        
        lines = []
        lines.append("API PERFORMANCE SUMMARY")
        lines.append("=" * 80)
        
        # Header
        header = f"{'API Call':<30} {'Count':<8} {'Avg(ms)':<8} {'Min(ms)':<8} {'Max(ms)':<8} {'Total(s)':<9} {'Cache Hit%':<10}"
        lines.append(header)
        lines.append("─" * len(header))
        
        # Sort by total time descending
        sorted_apis = sorted(
            self.api_stats.items(),
            key=lambda x: x[1].total_time,
            reverse=True
        )
        
        total_api_time = 0
        total_calls = 0
        total_cache_hits = 0
        total_cache_requests = 0
        
        for api_name, stats in sorted_apis:
            avg_ms = stats.avg_time * 1000
            min_ms = stats.min_time * 1000 if stats.min_time != float('inf') else 0
            max_ms = stats.max_time * 1000
            total_s = stats.total_time
            cache_pct = stats.cache_hit_rate
            
            line = f"{api_name:<30} {stats.count:<8} {avg_ms:<8.1f} {min_ms:<8.1f} {max_ms:<8.1f} {total_s:<9.2f} {cache_pct:<10.1f}"
            lines.append(line)
            
            total_api_time += stats.total_time
            total_calls += stats.count
            total_cache_hits += stats.cache_hits
            total_cache_requests += stats.cache_hits + stats.cache_misses
        
        lines.append("─" * len(header))
        
        # Summary statistics
        overall_cache_rate = (total_cache_hits / max(total_cache_requests, 1)) * 100
        runtime = time.time() - self.start_time
        
        lines.append(f"Total API Time: {total_api_time:.2f}s ({total_api_time/runtime*100:.1f}% of runtime)")
        lines.append(f"Total API Calls: {total_calls}")
        lines.append(f"Overall Cache Hit Rate: {overall_cache_rate:.1f}%")
        lines.append(f"Runtime: {runtime:.2f}s")
        
        # Slowest calls
        lines.append("\nSLOWEST API CALLS:")
        lines.append("─" * 40)
        
        all_slow_calls = []
        for api_name, stats in self.api_stats.items():
            for call in stats.slowest_calls:
                all_slow_calls.append((api_name, call))
        
        # Sort by duration and take top 10
        all_slow_calls.sort(key=lambda x: x[1]['duration'], reverse=True)
        
        for i, (api_name, call) in enumerate(all_slow_calls[:10], 1):
            duration_ms = call['duration'] * 1000
            context = call['context']
            lines.append(f"{i:2}. {api_name:<25} {duration_ms:6.1f}ms - {context}")
        
        return "\n".join(lines)
    
    def get_metrics_dict(self) -> Dict[str, Any]:
        """
        Get performance metrics as a dictionary for programmatic access.
        
        Returns:
            Dictionary containing performance metrics
        """
        metrics = {
            'api_stats': {},
            'summary': {
                'total_api_time': sum(stats.total_time for stats in self.api_stats.values()),
                'total_calls': sum(stats.count for stats in self.api_stats.values()),
                'runtime': time.time() - self.start_time,
            }
        }
        
        for api_name, stats in self.api_stats.items():
            metrics['api_stats'][api_name] = {
                'count': stats.count,
                'total_time': stats.total_time,
                'avg_time': stats.avg_time,
                'min_time': stats.min_time if stats.min_time != float('inf') else 0,
                'max_time': stats.max_time,
                'cache_hit_rate': stats.cache_hit_rate,
                'slowest_calls': stats.slowest_calls
            }
        
        return metrics
    
    def reset(self):
        """Reset all performance statistics."""
        self.api_stats.clear()
        self.start_time = time.time()
        self.logger.info("Performance monitor reset")