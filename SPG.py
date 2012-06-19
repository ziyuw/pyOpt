import numpy as np
"""
NOTE: 	1. isLegal is not implemented
		2. Cubic line search is not implemented (Currently only halving)
"""

def SPG(funObj, funProj, x, maxIter=200, verbose=2, suffDec=1e-4, progTol=1e-9, optTol=1e-7, curvilinear=0):
	"""
		Options:
			1. verbose: level of verbosity (0: no output, 1: final, 2: iter (default), 3:debug)
			2. optTol: tolerance used to check for optimality (default: 1e-5)
			3. progTol: tolerance used to check for lack of progress (default: 1e-9)
			4. maxIter: maximum number of calls to funObj (default: 500)
			5. suffDec: sufficient decrease parameter in Armijo condition (default: 1e-4)
			6. curvilinear: backtrack along projection Arc (default: 0)
	"""
	
	# Initilization
	testOpt = 1
	memory = 10
	
	x = funProj(x)
	
	f,g = funObj(x)
	projects = 1
	funEvals = 1
	funEvalMultiplier = 1
	i = 1
	
	while funEvals <= maxIter:
		if i == 1:
			alpha = 1
		else:
			y = g-g_old
			s = x-x_old
			
			alpha = np.dot(s.T, s)/np.dot(s.T, y)
			
			if alpha <= 1e-10 or alpha > 1e10:
				alpha = 1
			
		d = -alpha*g
		f_old = f
		x_old = x
		g_old = g
		
		if not curvilinear:
			d = funProj(x + d) - x
			projects = projects + 1

		gtd = np.dot(g.T, d)
		
		if gtd > -progTol:
			if verbose >= 1:
				print 'Directional Derivative below progTol'
			break
		
		if i == 1:
			t = np.min(1, 1/np.sum(np.absolute(g)))
		else:
			t = 1
	
		if memory == 1:
			funRef = f
		else:
			if i == 1:
				old_fvals = np.tile(-np.inf,(memory, 1))

			if i <= memory:
				old_fvals[i-1] = f
			else:
				old_fvals = np.vstack([old_fvals[1:],f])
			
			funRef = np.max(old_fvals)
		
		if curvilinear:
			x_new = funProj(x + t*d)
			projects = projects+1
		else:
			x_new = x + t*d
		
		f_new, g_new = funObj(x_new)
		funEvals = funEvals + 1
		
		lineSearchIters = 1;
		while f_new > funRef + suffDec*np.dot(g.T, (x_new-x)):
			temp = t;
			
			# Halfing step size
			t = t/2
			
			
			if t < temp*1e-3:
				if verbose == 3:
					print 'Interpolated value too small, Adjusting'
				t = temp*1e-3
				
			elif t > temp*0.6:
				if verbose == 3:
					print 'Interpolated value too large, Adjusting'
				t = temp*0.6
				

			# Check whether step has become too small
			if np.max(np.absolute(t*d)) < progTol or t == 0:
				if verbose == 3:
					print 'Line Search failed'
					
				t = 0
				f_new = f
				g_new = g
				break

			# Evaluate New Point
			f_prev = f_new
			t_prev = temp
			
			if curvilinear:
				x_new = funProj(x + t*d)
				projects = projects+1
			else:
				x_new = x + t*d
			
			f_new, g_new = funObj(x_new)
			funEvals = funEvals+1
			lineSearchIters = lineSearchIters+1
		
		# Take Step
		x = x_new
		f = f_new
		g = g_new
		
		if testOpt:
			optCond = np.max(np.absolute(funProj(x-g)-x));
			projects = projects+1;

		# Output Log
		if verbose >= 2:
			if testOpt:
				print '{:10d} {:10d} {:10d} {:15.5e} {:15.5e} {:15.5e}'.format(i, funEvals*funEvalMultiplier, projects, t, f, optCond)
			else:
				print '{:10d} {:10d} {:10d} {:15.5e} {:15.5e}'.format(i, funEvals*funEvalMultiplier, projects, t, f)
		
		
		# Check optimality
		if testOpt:
			if optCond < optTol:
				if verbose >= 1:
					print 'First-Order Optimality Conditions Below optTol'
				break
		
		if np.max(np.absolute(t*d)) < progTol:
			if verbose >= 1:
				print 'Step size below progTol'
			break
		

		if np.absolute(f-f_old) < progTol:
			if verbose >= 1:
				print 'Function value changing by less than progTol'
			break
		

		if funEvals*funEvalMultiplier > maxIter:
			if verbose >= 1:
				print 'Function Evaluations exceeds maxIter'
			break

		i = i + 1
	
	return x, f

if __name__ == '__main__':
	from numpy.linalg import *

	funcObj = lambda x: (0.5*np.dot(x.T, x) + np.dot(np.array([2,-2]), x), np.array([2,-2]).T + x)
	
	def proj(x):
		if norm(x) > 1:
			return x/norm(x)
		else:
			return x
	
	init_point = np.array([1,2]).T
	x, f = SPG(funcObj, proj, init_point, curvilinear=1)
	
	print x, f