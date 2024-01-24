import numpy as np
import mne
from scipy.io import loadmat


class BNCI(object):
    name_to_code_train = {'1536': 1, '1537': 2, '1538': 3, '1539': 4, '1540': 5, '1541': 6, '1542': 7, '33536': 8,
                          '33554': 9, '34304': 10,
                          '34305': 11, '34306': 12, '34307': 13, '34308': 14, '34309': 15, '34310': 16, '768': 17,
                          '785': 18, '786': 19}

    def __init__(self, filename, load_sensor_names=None, labels_filename=None):
        assert load_sensor_names is None
        self.__dict__.update(locals())
        del self.self

    def load(self):
        cnt = self.extract_data()
        events, artifact_trial_mask = self.extract_events(cnt)
        cnt.info["events"] = events
        cnt.info["artifact_trial_mask"] = artifact_trial_mask
        return cnt

    def extract_data(self):
        raw_gdf = mne.io.read_raw_gdf(self.filename, stim_channel="auto")
        raw_gdf.load_data()
        # correct nan values

        data = raw_gdf.get_data()

        for i_chan in range(data.shape[0]):
            # first set to nan, than replace nans by nanmean.
            this_chan = data[i_chan]
            data[i_chan] = np.where(
                this_chan == np.min(this_chan), np.nan, this_chan
            )
            mask = np.isnan(data[i_chan])
            chan_mean = np.nanmean(data[i_chan])
            data[i_chan, mask] = chan_mean

        gdf_events = mne.events_from_annotations(raw_gdf)
        gdf_events = self.pair_events(gdf_events)
        raw_gdf = mne.io.RawArray(data, raw_gdf.info, verbose="WARNING")
        # remember gdf events
        raw_gdf.info["gdf_events"] = gdf_events
        return raw_gdf

    def pair_events(self, gdf_events):
        events, name_to_code = gdf_events
        if not ("783" in name_to_code):
            if name_to_code == self.name_to_code_train:
                return gdf_events
            old2new = {}
            for key, idx in name_to_code.items():
                new_idx = self.name_to_code_train[key]
                old2new[idx] = new_idx
            for i, event in enumerate(events):
                event[2] = old2new[event[2]]
                gdf_events[0][i] = event
            return gdf_events
        else:
            if name_to_code == self.name_to_code_test:
                return gdf_events
            old2new = {}
            for key, idx in name_to_code.items():
                new_idx = self.name_to_code_test[key]
                old2new[idx] = new_idx
            gdf_events[1] = self.name_to_code_test
            for i, event in enumerate(events):
                event[2] = old2new[event[2]]
                gdf_events[0][i] = event
            return gdf_events

    def extract_events(self, raw_gdf):
        # all events
        events, name_to_code = raw_gdf.info["gdf_events"]

        if not ("783" in name_to_code):
            train_set = True
            assert all([s in name_to_code for s in ["1536", "1537", "1540", "1541"]])
        else:
            train_set = False
            assert ("783" in name_to_code)

        if train_set:
            trial_codes = [1, 2, 3, 4, 5, 6, 7]  # the 4 classes
        else:
            trial_codes = [8]  # "unknown" class

        trial_mask = np.array(
            [ev_code in trial_codes for ev_code in events[:, 2]])
        trial_events = np.array(events[trial_mask]).copy()
        assert len(trial_events) == 42, "Got {:d} markers".format(
            len(trial_events)
        )
        # from 1 2 5 6 to 1-4
        # trial_events[:, 2] = list(map(lambda a: a - 2 if a > 2 else a, trial_events[:, 2]))
        # possibly overwrite with markers from labels file
        if self.labels_filename is not None:
            classes = loadmat(self.labels_filename)["classlabel"].squeeze()
            if train_set:
                np.testing.assert_array_equal(trial_events[:, 2], classes)
            trial_events[:, 2] = classes
        unique_classes = np.unique(trial_events[:, 2])
        assert np.array_equal(
            [1, 2, 3, 4, 5, 6, 7], unique_classes
        ), "Expect 1,2,3,4,5,6,7 as class labels, got {:s}".format(
            str(unique_classes)
        )

        # now also create 0-1 vector for rejected trials
        artifact_trial_mask = np.zeros(len(trial_events), dtype=np.uint8)
        return trial_events, artifact_trial_mask
