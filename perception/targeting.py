"""Target selection shared by the ROS vision node and the simulator demo."""


def choose_target(results, threshold=0.75):
    """results: [{'xyz', 'cls', 'unc'}]. Out-of-distribution objects win (they must trigger the e-stop);
    otherwise the most certain known object."""
    if not results:
        return None
    worst = max(results, key=lambda r: r['unc'])
    return worst if worst['unc'] > threshold else min(results, key=lambda r: r['unc'])
