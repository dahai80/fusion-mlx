"""
System monitoring utilities for MLX-GUI.
Handles memory checking, system resource monitoring, and hardware detection.
"""

import logging
import platform
import subprocess
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import psutil

logger = logging.getLogger(__name__)


@dataclass
class SystemMemory:
    """System memory information."""
    total_gb: float
    available_gb: float
    used_gb: float
    percent_used: float
    
    def can_load_model(self, required_gb: float, safety_margin: float = 0.1) -> bool:
        """Check if system can load a model requiring specified memory."""
        # Add safety margin (default 10%)
        required_with_margin = required_gb * (1 + safety_margin)
        return self.available_gb >= required_with_margin
    
    def get_load_recommendation(self, required_gb: float) -> str:
        """Get a human-readable recommendation for model loading."""
        if self.can_load_model(required_gb):
            return f"✅ Safe to load - {self.available_gb:.1f}GB available, {required_gb:.1f}GB required"
        else:
            return f"⚠️ Memory warning - {self.available_gb:.1f}GB available, {required_gb:.1f}GB required (may cause system slowdown)"


@dataclass
class GPUMemory:
    """GPU memory information (if available)."""
    total_gb: float
    used_gb: float
    available_gb: float
    percent_used: float
    device_name: str


@dataclass
class SystemInfo:
    """Complete system information."""
    platform: str
    architecture: str
    processor: str
    is_apple_silicon: bool
    memory: SystemMemory
    gpu_memory: Optional[GPUMemory]
    mlx_compatible: bool


class SystemMonitor:
    """System monitoring and resource checking."""
    
    def __init__(self):
        self._system_info: Optional[SystemInfo] = None
    
    def get_memory_info(self) -> SystemMemory:
        """Get current system memory information."""
        memory = psutil.virtual_memory()
        
        total_gb = memory.total / (1024**3)
        available_gb = memory.available / (1024**3)
        used_gb = memory.used / (1024**3)
        percent_used = memory.percent
        
        return SystemMemory(
            total_gb=total_gb,
            available_gb=available_gb,
            used_gb=used_gb,
            percent_used=percent_used
        )
    
    def get_gpu_memory_info(self) -> Optional[GPUMemory]:
        """Get GPU memory information if available."""
        try:
            # Try to get Apple Silicon GPU memory using system_profiler
            if self.is_apple_silicon():
                return self._get_apple_silicon_gpu_memory()
            else:
                # Could add NVIDIA/AMD GPU detection here
                return None
        except Exception as e:
            logger.debug(f"Could not get GPU memory info: {e}")
            return None
    
    def _get_apple_silicon_gpu_memory(self) -> Optional[GPUMemory]:
        """Get Apple Silicon GPU memory information."""
        try:
            # Apple Silicon uses unified memory, so GPU memory is shared with system memory
            # We'll use a heuristic based on the chip type
            chip_info = self._get_apple_chip_info()
            
            if chip_info:
                # Estimate GPU memory based on chip type
                # This is approximate since Apple Silicon uses unified memory
                total_memory = self.get_memory_info().total_gb
                
                if "M1" in chip_info or "M2" in chip_info:
                    # M1/M2 typically use about 25-30% of memory for GPU
                    gpu_memory_gb = total_memory * 0.3
                elif "M3" in chip_info:
                    # M3 may have better GPU memory allocation
                    gpu_memory_gb = total_memory * 0.35
                else:
                    # Default estimate
                    gpu_memory_gb = total_memory * 0.25
                
                return GPUMemory(
                    total_gb=gpu_memory_gb,
                    used_gb=0.0,  # Hard to determine on Apple Silicon
                    available_gb=gpu_memory_gb,
                    percent_used=0.0,
                    device_name=chip_info
                )
        except Exception as e:
            logger.debug(f"Could not get Apple Silicon GPU info: {e}")
            
        return None
    
    def _get_apple_chip_info(self) -> Optional[str]:
        """Get Apple chip information."""
        try:
            result = subprocess.run(
                ["system_profiler", "SPHardwareDataType"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'Chip:' in line:
                        return line.split('Chip:')[1].strip()
                    elif 'Processor Name:' in line and 'Apple' in line:
                        return line.split('Processor Name:')[1].strip()
        except Exception as e:
            logger.debug(f"Could not get Apple chip info: {e}")
        
        return None
    
    def is_apple_silicon(self) -> bool:
        """Check if running on Apple Silicon."""
        return (
            platform.system() == "Darwin" and 
            platform.machine() == "arm64"
        )
    
    def is_mlx_compatible(self) -> bool:
        """Check if system is compatible with MLX."""
        return self.is_apple_silicon()
    
    def get_system_info(self) -> SystemInfo:
        """Get complete system information."""
        if self._system_info is None:
            self._system_info = SystemInfo(
                platform=platform.system(),
                architecture=platform.machine(),
                processor=platform.processor(),
                is_apple_silicon=self.is_apple_silicon(),
                memory=self.get_memory_info(),
                gpu_memory=self.get_gpu_memory_info(),
                mlx_compatible=self.is_mlx_compatible()
            )
        
        return self._system_info
    
    def check_model_compatibility(self, required_memory_gb: float) -> Tuple[bool, str]:
        """
        Check if a model can be loaded on this system.
        
        Args:
            required_memory_gb: Memory required by the model in GB
            
        Returns:
            Tuple of (can_load, message) - now always returns True for memory but with warnings
        """
        system_info = self.get_system_info()
        
        # Check MLX compatibility first - this is still a hard requirement
        if not system_info.mlx_compatible:
            return False, "❌ MLX requires Apple Silicon (M1/M2/M3) hardware"
        
        # Check memory requirements based on total system RAM (80% usable for models)
        memory_info = system_info.memory
        max_model_memory = memory_info.total_gb * 0.8  # 80% of total RAM available for models
        
        if required_memory_gb > max_model_memory:
            return True, f"⚠️ Memory warning - This model requires {required_memory_gb:.1f}GB but only {max_model_memory:.1f}GB ({memory_info.total_gb:.1f}GB total * 80%) is typically available for models. Loading may cause system slowdown or instability."
        
        return True, f"✅ Compatible - {required_memory_gb:.1f}GB required, {max_model_memory:.1f}GB available ({memory_info.total_gb:.1f}GB total * 80%)"
    
    def get_system_summary(self) -> Dict:
        """Get a summary of system information for API responses."""
        info = self.get_system_info()
        
        return {
            "platform": info.platform,
            "architecture": info.architecture,
            "processor": info.processor,
            "is_apple_silicon": info.is_apple_silicon,
            "mlx_compatible": info.mlx_compatible,
            "memory": {
                "total_gb": round(info.memory.total_gb, 1),
                "available_gb": round(info.memory.available_gb, 1),
                "used_gb": round(info.memory.used_gb, 1),
                "percent_used": round(info.memory.percent_used, 1)
            },
            "gpu_memory": {
                "total_gb": round(info.gpu_memory.total_gb, 1),
                "available_gb": round(info.gpu_memory.available_gb, 1),
                "device_name": info.gpu_memory.device_name
            } if info.gpu_memory else None
        }
    
    def log_system_info(self):
        """Log system information for debugging."""
        info = self.get_system_info()
        
        logger.info(f"System: {info.platform} {info.architecture}")
        logger.info(f"Processor: {info.processor}")
        logger.info(f"Apple Silicon: {info.is_apple_silicon}")
        logger.info(f"MLX Compatible: {info.mlx_compatible}")
        logger.info(f"Memory: {info.memory.total_gb:.1f}GB total, {info.memory.available_gb:.1f}GB available")
        
        if info.gpu_memory:
            logger.info(f"GPU: {info.gpu_memory.device_name}, {info.gpu_memory.total_gb:.1f}GB")


# Global system monitor instance
_system_monitor: Optional[SystemMonitor] = None


def get_system_monitor() -> SystemMonitor:
    """Get the global system monitor instance."""
    global _system_monitor
    if _system_monitor is None:
        _system_monitor = SystemMonitor()
    return _system_monitor