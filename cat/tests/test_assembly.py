"""ROS-message boundaries must not define the duration of CAT's input history."""
import unittest
import numpy as np

from perception_core import LivoxPacket, ScanBuffer, livox_points


BASE = 1000000000
PERIOD = 100000000


def packet(start=BASE, count=96, duration=469965, endian='<', offsets=None):
    dtype = np.dtype(dict(names=['offset_time','x','y','z','reflectivity','tag','line'],
        formats=[endian+'u4',endian+'f4',endian+'f4',endian+'f4','u1','u1','u1'],
        offsets=[0,4,8,12,16,17,18], itemsize=20))
    records = np.zeros(count if offsets is None else len(offsets), dtype=dtype)
    records['offset_time'] = (np.linspace(0,duration,count).astype(np.uint32)
                              if offsets is None else offsets)
    records['x'], records['y'], records['z'] = .2, .3, .4
    return LivoxPacket(records, len(records), 'livox_frame', start,
                       start+int(records['offset_time'].max()))


class AssemblyTests(unittest.TestCase):
    def test_small_packets_are_assembled_and_survive_delayed_tf(self):
        buf = ScanBuffer()
        for i in range(2400):  # ~2 kHz packet input, not 2 kHz scans
            buf.append(packet(BASE+i*500000), 10+i*.0005)
        scans, generation = buf.snapshot(now=11.2)
        chosen = ScanBuffer.select(scans, 11.2, 1.5, -1, BASE+8*PERIOD)
        self.assertIsNotNone(chosen)  # old 16-packet queue held only 8 ms
        self.assertEqual(chosen.start_ns, BASE+7*PERIOD)
        self.assertEqual(chosen.message.point_num, 200*96)
        self.assertEqual(chosen.packet_count, 200)
        self.assertGreater(chosen.end_ns-chosen.start_ns, 99000000)
        self.assertEqual(generation, 0)
        self.assertGreater(buf.stats(11.2)['queued_sensor_span_s'], 1.)
        xyz, _, _ = livox_points(chosen.message)
        self.assertEqual(len(xyz), 200*96)

    def test_full_scan_messages_keep_their_density(self):
        buf = ScanBuffer()
        for i in range(3):
            buf.append(packet(BASE+i*PERIOD, 19968, PERIOD-1), 10+i*.1)
        scans, _ = buf.snapshot()
        self.assertEqual(len(scans), 2)
        self.assertTrue(all(s.message.point_num == 19968 for s in scans))
        self.assertTrue(all(s.packet_count == 1 for s in scans))

    def test_split_unsorted_bundled_points_preserves_absolute_times_and_input(self):
        for endian in ('<','>'):
            buf = ScanBuffer()
            expected = []
            original = []
            for i in range(4):
                p = packet(BASE+i*PERIOD, endian=endian,
                           offsets=[100100000,80000000,0,99999999,20000000])
                original.append((p.points, p.points.copy()))
                expected.extend(p.start_ns+p.points['offset_time'].astype(np.int64))
                buf.append(p, 10+i*.1)
            scans, _ = buf.snapshot()
            actual = np.concatenate([s.start_ns+s.message.points['offset_time'].astype(np.int64)
                                     for s in scans])
            expected = np.array([t for t in expected if t < BASE+3*PERIOD])
            np.testing.assert_array_equal(np.sort(actual), np.sort(expected))
            for after, before in original:
                np.testing.assert_array_equal(after, before)
            self.assertTrue(all(not s.message.points.flags.writeable for s in scans))

    def test_switch_between_packet_and_scan_formats(self):
        buf = ScanBuffer()
        for i in range(200):
            buf.append(packet(BASE+i*500000), 10+i*.0005)
        buf.append(packet(BASE+PERIOD, 19968, PERIOD-1), 10.1)
        for i in range(201):
            buf.append(packet(BASE+2*PERIOD+i*500000), 10.2+i*.0005)
        scans, _ = buf.snapshot()
        self.assertEqual([s.message.point_num for s in scans], [19200,19968,19200])

    def test_overlapping_short_packet_does_not_invent_a_gap_in_a_long_scan(self):
        buf = ScanBuffer()
        buf.append(packet(BASE, 400, 2*PERIOD-1), 10.)
        buf.append(packet(BASE+PERIOD), 10.1)
        buf.append(packet(BASE+2*PERIOD), 10.2)
        scans, _ = buf.snapshot()
        self.assertEqual(len(scans), 2)
        self.assertEqual(sum(s.message.point_num for s in scans), 496)

    def test_oversized_assembled_scan_is_rejected_before_point_processing(self):
        buf = ScanBuffer()
        buf.append(packet(BASE, 60000, 40000000), 10.)
        buf.append(packet(BASE+50000000, 60000, 40000000), 10.05)
        buf.append(packet(BASE+PERIOD), 10.1)
        self.assertEqual(buf.snapshot()[0], [])
        self.assertEqual(buf.stats(10.1)['discarded_partial_windows'], 1)

    def test_oldest_receive_time_is_not_refreshed_by_later_parts(self):
        buf = ScanBuffer()
        for i in range(201):
            buf.append(packet(BASE+i*500000), 10+i*.0005)
        scan = buf.snapshot()[0][0]
        self.assertEqual(scan.received, 10.)
        self.assertIsNone(ScanBuffer.select([scan], 11.51, 1.5, -1, BASE+PERIOD))

    def test_history_expires_by_time_even_if_no_input_arrives(self):
        buf = ScanBuffer()
        for i in range(30):
            buf.append(packet(BASE+i*PERIOD, 100, PERIOD-1), 10+i*.1)
        scans, _ = buf.snapshot(now=12.9)
        self.assertTrue(all(12.9-s.received <= 1.5 for s in scans))
        self.assertGreater(len(scans), 10)
        self.assertEqual(buf.snapshot(now=15)[0], [])
        self.assertEqual(buf.stats(15)['buffered_points'], 0)

    def test_duplicates_do_not_refresh_receive_age(self):
        buf = ScanBuffer()
        buf.append(packet(count=100, duration=PERIOD-1), 10.)
        p = packet(BASE+PERIOD, 100, PERIOD-1)
        buf.append(p, 10.1)
        buf.append(p, 100.)
        self.assertEqual(buf.stats(100.)['latest_receive_age_s'], 89.9)
        self.assertEqual(buf.stats(100.)['duplicate_packets'], 1)
        self.assertEqual(buf.snapshot(now=100)[0], [])

    def test_clock_reset_discards_completed_and_partial_history(self):
        buf = ScanBuffer()
        buf.append(packet(count=100, duration=PERIOD-1), 10.)
        buf.append(packet(BASE+PERIOD, 100, PERIOD-1), 10.1)
        self.assertEqual(len(buf.snapshot()[0]), 1)
        buf.append(packet(BASE-1), 10.2)
        scans, generation = buf.snapshot()
        self.assertEqual(scans, [])
        self.assertEqual(generation, 1)
        self.assertEqual(buf.stats(10.2)['buffered_points'], 96)

    def test_data_gap_does_not_complete_a_tiny_fragment(self):
        buf = ScanBuffer()
        buf.append(packet(), 10.)
        buf.append(packet(BASE+2*PERIOD), 10.2)
        self.assertEqual(buf.snapshot()[0], [])
        self.assertEqual(buf.stats(10.2)['discarded_partial_windows'], 1)

    def test_receive_stall_does_not_rejuvenate_partial_window(self):
        buf = ScanBuffer()
        buf.append(packet(count=100, duration=PERIOD-1), 10.)
        buf.append(packet(BASE+PERIOD, 100, PERIOD-1), 15.)
        self.assertEqual(buf.snapshot()[0], [])

    def test_pending_point_budget_resets_atomically(self):
        buf = ScanBuffer(max_points=100)
        buf.append(packet(), 10.)
        with self.assertRaisesRegex(ValueError,'memory budget'):
            buf.append(packet(BASE+500000), 10.0005)
        self.assertEqual(buf.snapshot(), ([],1))
        self.assertEqual(buf.stats(10.1)['buffered_points'], 0)
        self.assertEqual(buf.stats(10.1)['budget_resets'], 1)

    def test_pending_object_budget_is_bounded_for_tiny_packets(self):
        buf = ScanBuffer(max_parts=2)
        for i in range(2):
            buf.append(packet(BASE+i*1000, count=1, duration=0), 10.)
        with self.assertRaisesRegex(ValueError,'memory budget'):
            buf.append(packet(BASE+2000, count=1, duration=0), 10.)
        self.assertEqual(buf.stats(10.)['buffered_points'], 0)

    def test_completed_scan_count_and_point_budgets(self):
        for options in (dict(max_scans=2), dict(max_points=300)):
            buf = ScanBuffer(**options)
            for i in range(6):
                buf.append(packet(BASE+i*PERIOD, count=100, duration=PERIOD-1), 10+i*.1)
            self.assertEqual(len(buf.snapshot()[0]), 2)
            self.assertLessEqual(buf.stats(10.6)['buffered_points'], 300)


if __name__ == '__main__':
    unittest.main()
