function disjoint = disjointSet(polyhedra)
%DISJOINTSET Returns true if there are no overlapping polyhedra
disjoint = ~PolyUnion(polyhedra).isOverlapping();

end

