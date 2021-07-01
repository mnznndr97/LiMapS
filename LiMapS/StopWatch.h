#pragma once

#include <chrono>

class StopWatch {
	std::chrono::steady_clock::time_point _start;
	std::chrono::steady_clock::time_point _end;
	bool _isRunning;

	double _elapsed;
public:
	inline StopWatch() {
		_isRunning = false;
		_elapsed = 0;
	}

	inline void Restart() {
		_elapsed = 0;
		_start = std::chrono::high_resolution_clock::now();
		_isRunning = true;
	}

	inline void Stop() {
		_end = std::chrono::high_resolution_clock::now();
		_elapsed = std::chrono::duration<double, std::milli>(_end - _start).count();
		_isRunning = false;
	}

	inline double Elapsed() const {
		double result = _elapsed;

		if (_isRunning) {
			std::chrono::steady_clock::time_point current = std::chrono::high_resolution_clock::now();
			result = std::chrono::duration<double, std::milli>(current - _start).count();
		}

		return result;
	}
};
