(defun gen-bn (nodes max-parents  
		     &optional (polytree t) (ratio-uc .3) 
		      &aux (uc-nodes (max 2 (floor (* ratio-uc nodes))))
		      (bn nil))
   ;; Generates a bayesian network (a polytree if argument is set to T)
   ;; You can choose the number of nodes, the max number of parents per node,
   ;; and the ratio of independent nodes
   (format t "Nodes without parents ~d~%" uc-nodes)
   (dotimes (i uc-nodes bn)
	    (push (list i nil nil nil (list (random 1.0))) bn))
   (dotimes (i (- nodes uc-nodes)); (reverse bn)
	    (push (gen-node (+ i uc-nodes) max-parents bn polytree) bn))
   (print-bn (reverse bn))
   (reverse bn))

(defun gen-node (id max-parents bn polytree &aux 
		    (num-parents (max 1 (random (min (1+ max-parents) id)))))
		    ;num-parents should be >=1
		    ;(num-parents (random (min (1+ max-parents) id)))
  ;;Generates a node with # parents <= max-parents randomly selected from existing nodes
  (let ((parents-ancestors (gen-parents id num-parents bn polytree)))
    (list id (first parents-ancestors) (second parents-ancestors) nil
	  (if (null (first parents-ancestors)) ; if no parents could be selected
	      (list (random 1.0)) ; e.g., any selection would violate polytree constraint
	    (gen-cpt (length (first parents-ancestors)))))))

(defun gen-parents (id num-parents bn polytree &aux (parents nil) (ancestors nil)
		       (candidates (do* ((i 0 (1+ i)) (l (list 0) (push i l)))
					((= i (1- id)) l))))
  ;;Picks num-parents number of parents from existing nodes
  ;; returns pair of (parents ancestors)
  (do ((j 0 (1+ j)))
      ((or (= j num-parents) (null candidates))
       (list (sort parents #'<) (sort ancestors #'<)))
      (let ((parent (nth (random (length candidates)) candidates)))
	(setf candidates (delete parent candidates))
	(unless (and polytree ; (ancestor parent id bn)
		     (check-connected id parent ancestors bn))
	  (add-descendants id parent bn)
	  (dolist (node (ancestors parent bn))
	    	  (add-descendants id node bn))
;	  (dolist (node (ancestors parent bn))
;	    	  (push id (desendants node bn)))
	  (push parent parents)
	  (setf ancestors 
		(cons parent (append (ancestors parent bn) ancestors)))))))

(defun ancestors (id bn)  ;; returns the ancestors of node id in the bn
  (third (assoc id bn)))

(defun descendants (id bn)  ;; returns the ancestors of node id in the bn
  (fourth (assoc id bn)))

(defun prob-table (node) ;; returns the probabilities associaded with node
  (fifth node))

(defun add-descendants (id ancestor bn)
  (let ((parent-node (assoc ancestor bn)))
    (push id (fourth parent-node))))

(defun check-connected (id candidate-parent ancestors bn
			   &aux (candidate+ancestors (cons candidate-parent 
							   (ancestors candidate-parent bn))))
  (or 
   (intersection ancestors candidate+ancestors)
   (do* ((i 0 (1+ i))
	 (c+a (nth i candidate+ancestors) (nth i candidate+ancestors))
	 (ancestor-descendants (descendants c+a bn) 
			       (union ancestor-descendants (descendants c+a bn))))
	((= i (1- (length candidate+ancestors)))
	 (dolist (node ancestor-descendants nil)
	   (when (intersection (ancestors node bn) ancestors)
	     (return T)))))))
;   (dolist (c+a candidate+ancestors nil)
;     (when 
;	 (dolist (node (descendants c+a bn) nil)
;	   (when (intersection (ancestors node bn) ancestors)
;	     (return T)))
;       (return T)))))
       

(defun gen-cpt (num-parents &aux (cpt nil) (flag nil)
			    (row (make-list num-parents :initial-element 'F)))
  ;; Function that generates a cpt for the specificied number of parents
  (push (append (copy-alist row) (list (random 1.0))) cpt)
  (loop
   (do ((i 0 (1+ i)))
       ((or (= i num-parents) (eq (nth i row) 'F))
	(when (< i num-parents)
	  (setf flag t)
	  (setf (nth i row) 'T) 
	  (push (append (copy-alist row) (list (random 1.0))) cpt)))
       (setf (nth i row) 'F))
   (if flag
       (setf flag nil)
     (return cpt))))

(defun print-bn (bn &aux
		    (outfile (open "bn.data" :direction :output 
				   :if-does-not-exist :create 
				   :if-exists :supersede)))
  (dolist (node bn)
	  (format outfile "~a " (first node))
	  (dolist (parent (second node))
		  (format outfile "~a " parent))
					;(format outfile "~a ~%" (third node))
	  (format outfile "~%")
	  (if (null (second node))
	      (format outfile "~f~%" (first (prob-table node)))
	    (dolist (row (prob-table node))
		    (dolist (element row)
			    (format outfile "~a " element))
		    (format outfile "~%"))))
  (close outfile))
