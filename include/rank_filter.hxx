#ifndef __RANK_FILTER__
#define __RANK_FILTER__


#include <deque>
#include <cassert>
#include <functional>
#include <iostream>
#include <iterator>
#include <utility>

#include <boost/array.hpp>
#include <boost/container/set.hpp>
#include <boost/container/node_allocator.hpp>
#include <boost/math/special_functions/round.hpp>
#include <boost/static_assert.hpp>
#include <boost/type_traits.hpp>


namespace rank_filter
{

template<class I1,
         class I2>
inline void lineRankOrderFilter1D(const I1& src_begin, const I1& src_end,
        I2& dest_begin, I2& dest_end,
        size_t half_length, double rank)
{
    // Types in use.
    typedef typename std::iterator_traits<I1>::value_type T1;
    typedef typename std::iterator_traits<I2>::value_type T2;

    // Establish common types to work with source and destination values.
    BOOST_STATIC_ASSERT((boost::is_same<T1, T2>::value));
    typedef T1 T;

    // Define container types that will be used.
    typedef std::pair<T, size_t> TC;
    typedef boost::container::set< TC,
            std::less<TC>,
            boost::container::node_allocator<TC>,
            boost::container::tree_assoc_options< boost::container::tree_type<boost::container::scapegoat_tree> >::type> set;
    typedef std::deque< typename set::iterator > deque;

    // Precompute some window lengths.
    const size_t half_length_add_1 = half_length + 1;
    const size_t length_sub_1 = 2 * half_length;
    const size_t length = length_sub_1 + 1;

    // Ensure the result will fit.
    assert(std::distance(src_begin, src_end) <= std::distance(dest_begin, dest_end));

    // Window length cannot exceed input data with reflection.
    assert(half_length_add_1 <= std::distance(src_begin, src_end));

    // Rank must be in the range 0 to 1.
    assert((0 <= rank) && (rank <= 1));

    // Track values in window both in sorted and sequential order.
    size_t values_seen = 0;
    set sorted_window;
    deque window_iters(length);

    // Iterators in source and destination.
    I1 src_pos = src_begin;
    I1 dest_pos = dest_begin;

    // Get the initial window in sequential order with reflection.
    // Insert elements in order to the set for sorting.
    // Store all set iterators into the deque in sequential order.
    {
        std::deque<T> window_init(half_length_add_1);

        window_init[half_length] = *src_pos;
        for (size_t j = half_length; j > 0;)
        {
            window_init[--j] = *(++src_pos);
        }
        ++src_pos;

        TC each_pair;
        for (size_t j = 0; j < half_length; ++j, ++values_seen)
        {
            each_pair.first = window_init[j];
            each_pair.second = values_seen;
            window_iters[j] = sorted_window.insert(each_pair).first;
        }
        for (size_t j = half_length; j < length; ++j, ++values_seen)
        {
            each_pair.first = window_init.back();
            each_pair.second = values_seen;
            window_iters[j] = sorted_window.insert(each_pair).first;
            window_init.pop_back();
        }
    }

    // Window position corresponding to this rank.
    const size_t rank_pos = static_cast<size_t>(boost::math::round(rank * length_sub_1));
    typename set::iterator rank_point = sorted_window.begin();
    std::advance(rank_point, rank_pos);

    // Roll window forward one value at a time.
    typename set::iterator prev_iter;
    T prev_value;
    T rank_value = rank_point->first;
    *dest_pos = rank_value;
    T next_value;
    TC next_pair;
    size_t window_reflect_pos = length_sub_1;
    bool src_not_empty = ( src_pos != src_end );
    while ( src_not_empty || ( window_reflect_pos > 0 ) )
    {
        // Get last value to remove from window.
        prev_iter = window_iters.front();
        prev_value = prev_iter->first;
        window_iters.pop_front();

        // Determine next value to add to window.
        // Handle special cases like reflection at the end.
        if ( src_not_empty )
        {
            next_value = *src_pos;
            ++src_pos;
            src_not_empty = ( src_pos != src_end );
        }
        else
        {
            window_reflect_pos -= 2;
            next_value = window_iters[window_reflect_pos]->first;
        }
        next_pair.first = next_value;
        next_pair.second = ++values_seen;

        // Remove old value and add new value to the window.
        // Handle special cases where `rank_pos` may have an adjusted position
        // due to where the old and new values are inserted.
        if ( ( rank_value < prev_value ) && ( rank_value <= next_value ) )
        {
            sorted_window.erase(prev_iter);
            window_iters.push_back(sorted_window.insert(next_pair).first);
        }
        else if ( ( rank_value >= prev_value ) && ( rank_value > next_value ) )
        {
            if ( rank_point == prev_iter )
            {
                window_iters.push_back(sorted_window.insert(next_pair).first);
                --rank_point;
                sorted_window.erase(prev_iter);
            }
            else
            {
                sorted_window.erase(prev_iter);
                window_iters.push_back(sorted_window.insert(next_pair).first);
            }
        }
        else if ( ( rank_value < prev_value ) && ( rank_value > next_value ) )
        {
            sorted_window.erase(prev_iter);
            window_iters.push_back(sorted_window.insert(next_pair).first);
            --rank_point;
        }
        else if ( ( rank_value >= prev_value ) && ( rank_value <= next_value ) )
        {
            if (rank_point == prev_iter)
            {
                window_iters.push_back(sorted_window.insert(next_pair).first);
                ++rank_point;
                sorted_window.erase(prev_iter);
            }
            else
            {
                sorted_window.erase(prev_iter);
                window_iters.push_back(sorted_window.insert(next_pair).first);
                ++rank_point;
            }
        }

        // Store rank value in the destination.
        *(++dest_pos) = rank_value = (rank_point->first);
    }
}

}


namespace std
{
    template <class T, size_t N>
    ostream& operator<<(ostream& out, const boost::array<T, N>& that)
    {
        out << "{ ";
        for (unsigned int i = 0; i < (N - 1); ++i)
        {
            out << that[i] << ", ";
        }
        out << that[N - 1] << " }";

        return(out);
    }
}


#endif //__RANK_FILTER__
