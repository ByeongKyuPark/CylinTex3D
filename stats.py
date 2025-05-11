import pstats
# p = pstats.Stats('profile_stats.prof')
p = pstats.Stats('profile_stats_abc.prof')
p.sort_stats('cumulative').print_stats(20)