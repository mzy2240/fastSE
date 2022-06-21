
import contextlib
import os
from typing import List, AnyStr, Dict
import numpy as np
import pandas as pd
import re
from io import StringIO, TextIOWrapper


def interpret_line(line, splitter=','):
    """
    Split text into arguments and parse each of them to an appropriate format (int, float or string)
    Args:
        line: text line
        splitter: value to split by
    Returns: list of arguments
    """
    parsed = []
    elms = line.split(splitter)

    for elm in elms:
        try:
            # try int
            el = int(elm)
        except ValueError as ex1:
            try:
                # try float
                el = float(elm)
            except ValueError as ex2:
                # otherwise just leave it as string
                el = elm.strip()
        parsed.append(el)
    return parsed


def find_between(s, start, end):
    # if start in s and end in s:
    #     return (s.split(start))[1].split(end)[0]
    # else:
    #     return ""
    return (s.split(start))[1].split(end)[0]


def split_line(lne):

    chunks = []

    if len(lne):

        current = ''
        started = False
        reading_str = False
        float_detected = False
        for chr in lne:
            if chr != ':':

                if chr != ' ' and not started:
                    # start any
                    reading_str = chr == '"'
                    started = True
                    float_detected = False
                    current += chr

                elif chr not in [' ', '"'] and started and not reading_str:
                    # keep reading value

                    if not float_detected and chr == '.':
                        float_detected = True

                    current += chr

                elif chr == ' ' and started and not reading_str:
                    # finalize reading value
                    started = False
                    if float_detected:
                        chunks.append(float(current))
                    else:
                        try:
                            chunks.append(int(current))
                        except ValueError:
                            chunks.append(current)
                    current = ''

                elif chr != '"' and started and reading_str:
                    # keep reading string
                    current += chr

                elif chr == '"' and started and reading_str:
                    # finalize reading string
                    current += chr
                    started = False
                    chunks.append(current.replace('"', ''))
                    current = ''

        if len(current):
            if reading_str:
                chunks.append(current)
            elif float_detected:
                chunks.append(float(current))
            else:
                try:
                    chunks.append(int(current))
                except ValueError:
                    chunks.append(current)

    return chunks


class EPCReader:

    def __init__(self, file_name):
        """
        Parse PowerWorld EPC file
        Args:
            file_name: file name or path
        """
        self.parsers = {}
        self.versions = []
        self.file_name = file_name
        self.data_dict = None


    def read_and_split(self) -> (List[AnyStr], Dict[AnyStr, AnyStr]):
        """
        Read the text file and split it into sections
        :return: list of sections, dictionary of sections by type
        """

        # open the text file into a variable
        txt = ''
        try:
            with open(self.file_name, 'r', encoding='utf-8') as my_file:
                for line in my_file:
                    if line[0] != '@':
                        txt += line
        except TypeError:
            my_file = TextIOWrapper(self.file_name)
            for line in my_file:
                with contextlib.suppress(IndexError):
                    if line[0] != '@':
                        txt += line
            my_file.close()  # clear the space in memory
        # fix stupid line partition
        txt = txt.replace('/\n', '')

        expected_sections = ['title',
                             'comments',
                             'solution parameters',
                             'substation data',
                             'bus data',
                             'branch data',
                             'transformer data',
                             'generator data',
                             'load data',
                             'shunt data',
                             'svd data',
                             'area data',
                             'zone data',
                             'interface data',
                             'interface branch data',
                             'dc bus data',
                             'dc line data',
                             'dc converter data',
                             'vs converter data',
                             'z table data',
                             'gcd data',
                             'transaction data',
                             'owner data',
                             'qtable data',
                             'ba data',
                             'injgroup data',
                             'injgrpelem data',
                             'end']

        # find which of the expected sections are actually in the file
        present_sections = [a for a in expected_sections if a in txt]
        # split the text file into the found sections
        sections_dict = {}
        for i in range(len(present_sections)-1):
            a = present_sections[i]
            b = present_sections[i + 1]
            if a in txt and b in txt:
                raw_txt = find_between(txt, a, b)
                lines = raw_txt.split('\n')

                if len(lines) > 0:
                    if '[' in lines[0]:
                        new_lines = []
                        header = lines[0].split(']')[1].split()
                        for j in range(1, len(lines)):
                            line_data = split_line(lines[j])
                            if len(line_data) > 0:
                                new_lines.append(line_data)

                        sections_dict[a] = {'header': header, 'data': new_lines}
                    else:
                        sections_dict[a] = {'header': None, 'data': lines}
                else:
                    sections_dict[a] = {'header': '', 'data': lines}
            else:
                sections_dict[a] = {'header': '', 'data': list()}

        self.data_dict = sections_dict
        return sections_dict

    def convert(self, data_dict=None):
        if data_dict is None:
            data_dict = self.data_dict
        base = [s for s in data_dict['solution parameters']['data'] if "sbase" in s][0]
        base = float(re.findall("\d+\.\d+", base)[0])
        bus = pd.DataFrame(data_dict['bus data']['data'])
        bus.rename({0: 'BusNum', 1: 'BusName', 2: 'BusNomVolt', 26: 'SubNum', 27: 'SubName', 6: 'BusPUVolt', 8: 'BusAngle', 11: 'BusVoltLimHigh', 12: 'BusVoltLimLow'}, axis='columns', inplace=True)
        bus['BusCat'] = 'PQ'
        bus['BusLoadMW'] = 0
        bus['BusLoadMVR'] = 0
        bus['BusGenMW'] = 0
        bus['BusGenMVR'] = 0
        bus['GenMVRMax'] = 0
        bus['GenMVRMin'] = 0
        bus['BusSS'] = 0
        bus['BusSSMW'] = 0
        load = pd.DataFrame(data_dict['load data']['data'])
        load.rename({0: 'BusNum', 5: 'LoadStatus', 6: 'LoadMW', 7: 'LoadMVR'}, axis='columns', inplace=True)
        # iterate over the load df and add the MW and MVR to the bus df
        # make sure the load status is closed
        for i in range(len(load)):
            if load.iloc[i]['LoadStatus'] == 1:
                bus.loc[bus['BusNum'] == load.loc[i, 'BusNum'], 'BusLoadMW'] += load.loc[i, 'LoadMW']
                bus.loc[bus['BusNum'] == load.loc[i, 'BusNum'], 'BusLoadMVR'] += load.loc[i, 'LoadMVR']
        gen = pd.DataFrame(data_dict['generator data']['data'])
        gen.rename({0: 'BusNum', 5: 'GenStatus', 13: 'GenMW', 14: 'GenMWMax', 15: 'GenMWMin', 16: 'GenMVR', 17: 'GenMVRMax', 18: 'GenMVRMin'}, axis='columns', inplace=True)
        # iterate over the gen df and add the MW and MVR to the bus df
        # make sure the gen status is closed
        for i in range(len(gen)):
            if gen.iloc[i]['GenStatus'] == 1:
                bus.loc[bus['BusNum'] == gen.loc[i, 'BusNum'], 'BusGenMW'] += gen.loc[i, 'GenMW']
                bus.loc[bus['BusNum'] == gen.loc[i, 'BusNum'], 'BusGenMVR'] += gen.loc[i, 'GenMVR']
                bus.loc[bus['BusNum'] == gen.loc[i, 'BusNum'], 'GenMVRMax'] += gen.loc[i, 'GenMVRMax']
                bus.loc[bus['BusNum'] == gen.loc[i, 'BusNum'], 'GenMVRMin'] += gen.loc[i, 'GenMVRMin']
                bus.loc[bus['BusNum'] == gen.loc[i, 'BusNum'], 'BusCat'] = 'PV'
        # select the first pv bus as the slack bus
        bus[bus['BusCat'] == 'PV'].iloc[0]['BusCat'] = 'Slack'
        svd = pd.DataFrame(data_dict['svd data']['data'])
        svd.rename({0: 'BusNum', 13: 'SSNMVR'}, axis='columns', inplace=True)
        svd['SSNMVR'] = svd['SSNMVR'] * base 
        for i in range(len(svd)):
            bus.loc[bus['BusNum'] == svd.loc[i, 'BusNum'], 'BusSS'] = svd.loc[i, 'SSNMVR']
        # compute the net injection
        bus['BusNetMW'] = bus['BusGenMW'] - bus['BusLoadMW']
        bus['BusNetMVR'] = bus['BusGenMVR'] + bus['BusSS'] - bus['BusLoadMVR']
        # now let's process branch
        branch = pd.DataFrame(data_dict['branch data']['data'])
        branch.rename({0: 'BusNum', 1: 'BusName', 3: 'BusNum:1', 4: 'BusName:1', 10: 'LineR', 11: 'LineX', 12: 'LineC', 13: 'LineLimMVA'}, axis='columns', inplace=True)
        branch['BranchDeviceType'] = 'Line'
        transformer = pd.DataFrame(data_dict['transformer data']['data'])
        transformer.rename({0: 'BusNum', 1: 'BusName', 3: 'BusNum:1', 4: 'BusName:1', 23: 'LineR', 24: 'LineX', 25: 'LineC', 35: 'LineLimMVA'}, axis='columns', inplace=True)
        transformer['BranchDeviceType'] = 'Transformer'
        # only keep the renamed columns and combine the branch and transformer dataframes together
        branch = branch[['BusNum', 'BusName', 'BusNum:1', 'BusName:1', 'BranchDeviceType', 'LineR', 'LineX', 'LineC', 'LineLimMVA']]
        transformer = transformer[['BusNum', 'BusName', 'BusNum:1', 'BusName:1', 'BranchDeviceType', 'LineR', 'LineX', 'LineC', 'LineLimMVA']]
        branch = pd.concat([branch, transformer])
        branch['LineTap'] = 1  # might be available in the branch df
        branch['LinePhase'] = 0  # same as above
        # now let's process substation and add the coordinates to buses
        sub = pd.DataFrame(data_dict['substation data']['data'])
        sub.rename({0: 'SubNum', 1: 'SubName', 2: 'Latitude', 3: 'Longitude'}, axis='columns', inplace=True)
        bus = pd.merge(bus, sub[['SubNum', 'Latitude', 'Longitude']], on='SubNum', how='left')
        # add lat and long to the branch df
        s_lon_f = []
        s_lat_f = []
        s_lon_t = []
        s_lat_t = []
        for a, b in zip(branch['BusNum'], branch['BusNum:1']):
            bus_a = bus[bus['BusNum'] == a]
            bus_b = bus[bus['BusNum'] == b]
            lon_1 = bus_a['Longitude'].values[0]
            lat_1 = bus_a['Latitude'].values[0]
            lon_2 = bus_b['Longitude'].values[0]
            lat_2 = bus_b['Latitude'].values[0]
            s_lon_f.append(lon_1)
            s_lat_f.append(lat_1)
            s_lon_t.append(lon_2)
            s_lat_t.append(lat_2)
        branch['Longitude'] = s_lon_f
        branch['Latitude'] = s_lat_f
        branch['Longitude:1'] = s_lon_t
        branch['Latitude:1'] = s_lat_t
        return base, bus, branch


if __name__ == '__main__':
    from fastse.tdpf import run_tdpf
    from scipy.sparse import csr_matrix
    import time
    reader = EPCReader(r"C:\Users\test\PWCase\ACTIVSg200.EPC")
    reader.read_and_split()
    base, df, branch = reader.convert()
    np.set_printoptions(precision=4, floatmode='fixed')
    df['BusNetMW'] = df['BusNetMW'].astype(float)
    df['BusNetMVR'] = df['BusNetMVR'].astype(float)
    df['BusGenMW'] = df['BusGenMW'].astype(float)
    df['BusGenMVR'] = df['BusGenMVR'].astype(float)
    df['BusLoadMW'] = df['BusLoadMW'].astype(float)
    df['BusLoadMVR'] = df['BusLoadMVR'].astype(float)
    df['GenMVRMax'] = df['GenMVRMax'].astype(float)
    df['GenMVRMin'] = df['GenMVRMin'].astype(float)
    df['BusSS'] = df['BusSS'].astype(float)
    df['BusSSMW'] = df['BusSSMW'].astype(float)
    df['BusNum'] = df['BusNum'].astype(int)
    df.fillna(0, inplace=True)
    branch['LineR'] = branch['LineR'].astype(float)
    branch['LineX'] = branch['LineX'].astype(float)
    branch['LineC'] = branch['LineC'].astype(float)
    branch['LineTap'] = branch['LineTap'].astype(float)
    branch['LinePhase'] = branch['LinePhase'].astype(float)
    branch['LineLimMVA'] = branch['LineLimMVA'].astype(float)

    nb = df.shape[0]
    nl = branch.shape[0]

    Ys = 1 / (branch['LineR'].to_numpy() + 1j * branch['LineX'].to_numpy())  # series admittance
    Bc = branch['LineC'].to_numpy()  # line charging susceptance
    tap = branch['LineTap'].to_numpy()
    shift = branch['LinePhase'].to_numpy()
    rates = branch['LineLimMVA'].to_numpy()
    tap = tap * np.exp(1j * np.pi / 180 * shift)
    Ytt = Ys + 1j * Bc / 2
    Yff = Ytt / (tap * np.conj(tap))
    Yft = - Ys / np.conj(tap)
    Ytf = - Ys / tap

    Ysh = (df['BusSSMW'].to_numpy() + 1j * df['BusSS'].to_numpy()) / base


    # lookup table for formatting bus numbers
    def loop_translate(a, d):
        n = np.ndarray(a.shape, dtype=int)
        for k in d:
            n[a == k] = d[k]
        return n


    d = dict()
    for index, value in df['BusNum'].items():
        d[value] = index
    f = branch['BusNum'].to_numpy(dtype=int).reshape(-1)
    f = loop_translate(f, d)
    t = branch['BusNum:1'].to_numpy(dtype=int).reshape(-1)
    t = loop_translate(t, d)
    ## connection matrix for line & from buses
    # print(nl, nb, f, t)
    Cf = csr_matrix((np.ones(nl), (range(nl), f)), (nl, nb))
    ## connection matrix for line & to buses
    Ct = csr_matrix((np.ones(nl), (range(nl), t)), (nl, nb))
    i = np.r_[range(nl), range(nl)]  ## double set of row indices
    Yf = csr_matrix((np.hstack([Yff.reshape(-1), Yft.reshape(-1)]), (i, np.hstack([f, t]))),
                    (nl, nb))
    Yt = csr_matrix((np.hstack([Ytf.reshape(-1), Ytt.reshape(-1)]), (i, np.hstack([f, t]))),
                    (nl, nb))
    Ybus = Cf.T * Yf + Ct.T * Yt + csr_matrix((Ysh, (range(nb), range(nb))), (nb, nb))
    # print(Ybus.toarray())
    # Ybus = sa.get_ybus()

    pv = df.index[df['BusCat'].str.contains("PV")].to_numpy(int)
    pq = df.index[df['BusCat'].str.contains("PQ")].to_numpy(int)  # include PQ with Var limit
    pv.sort()
    pq.sort()
    slack = df.index[df['BusCat'] == 'Slack'].to_numpy(int)

    # set up indexing for updating v
    pvpq = np.r_[pv, pq]
    npv = len(pv)
    npq = len(pq)
    # print(npv, npq)
    npvpq = npv + npq

    pvpq_lookup = np.zeros(np.max(Ybus.indices) + 1, dtype=int)
    pvpq_lookup[pvpq] = np.arange(npvpq)

    Sbus = (df['BusNetMW'].to_numpy(dtype=float) + df['BusNetMVR'].to_numpy(
        dtype=float) * 1j) / base
    V = df['BusPUVolt'].to_numpy(complex)

    x = branch['LineX'].to_numpy()
    r = branch['LineR'].to_numpy()
    c = branch['LineC'].to_numpy()
    tap = branch['LineTap'].to_numpy()
    shift = branch['LinePhase'].to_numpy()
    f = branch['BusNum'].to_numpy(dtype=int).reshape(-1)
    f = loop_translate(f, d)
    t = branch['BusNum:1'].to_numpy(dtype=int).reshape(-1)
    t = loop_translate(t, d)
    line_indexes = (branch['BranchDeviceType'] == 'Line').to_numpy()
    tran_indexes = (branch['BranchDeviceType'] == 'Transformer').to_numpy()
    num_lines = line_indexes.sum()

    tc0 = np.full(num_lines, 25)
    # tas = np.random.uniform(20, 40, num_lines)
    tas = np.random.uniform(-10, 10, nl)

    radiations = np.random.uniform(800, 1200, num_lines)
    winds = np.random.uniform(0.5, 3, num_lines)

    start = time.time()
    result = run_tdpf(tas, tc0, radiations, winds, V, npv, npq, r, x, c, tap, shift, f, t, i, nl,
                      nb, Sbus, pv, pq, pvpq, base, line_indexes, tran_indexes, rates, 1e-3, 50)

    print(result.temperature)
    # print(result.temperature)

