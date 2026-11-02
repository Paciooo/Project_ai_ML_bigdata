"""A scripted stand-in for ib_async.IB (no network, no brokerage connection).

Trades are given as (datetime UTC, price, size). reqHistoricalTicksAsync
returns up to `numberOfTicks` trades from `startDateTime`, then completes the
last second (as IBKR documents). `script` injects failures per call number:
  ("error", code, msg)  -> error event for the request id, empty answer
  ("timeout",)          -> never answers
  ("disconnect",)       -> raises ConnectionError and drops the connection
"""

import asyncio
from collections import namedtuple
from datetime import timedelta

Tick = namedtuple("Tick", "time price size exchange specialConditions tickAttribLast")
Attr = namedtuple("Attr", "pastLimit unreported")
Bar = namedtuple("Bar", "date open high low close volume")


class Event(list):
    def __iadd__(self, f):
        self.append(f)
        return self

    def emit(self, *a):
        for f in self:
            f(*a)


class Client:
    def __init__(self):
        self._reqIdSeq = 1

    def getReqId(self):
        self._reqIdSeq += 1
        return self._reqIdSeq - 1


class FakeIB:
    def __init__(self, trades, script=None, bars_script=None):
        self.trades = sorted(trades, key=lambda t: t[0])
        self.script = dict(script or {})
        self.bars_script = dict(bars_script or {})
        self.errorEvent = Event()
        self.client = Client()
        self.connected = True
        self.n_tick_calls = 0
        self.n_bar_calls = 0
        self.connects = 0

    # connection
    def isConnected(self):
        return self.connected

    def connect(self, **kw):
        self.connected = True
        self.connects += 1

    def disconnect(self):
        self.connected = False

    def sleep(self, s):
        pass

    def run(self, aw):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(aw)
        finally:
            loop.close()

    async def _fail(self, action, rid):
        if action[0] == "error":
            self.errorEvent.emit(rid, action[1], action[2], None)
            return []
        if action[0] == "timeout":
            await asyncio.sleep(3600)
        if action[0] == "disconnect":
            self.connected = False
            raise ConnectionError("Socket disconnect")

    async def reqHistoricalTicksAsync(self, contract, startDateTime, endDateTime, numberOfTicks,
                                      whatToShow, useRth, **kw):
        rid = self.client.getReqId()
        self.n_tick_calls += 1
        action = self.script.get(self.n_tick_calls)
        if action:
            return await self._fail(action, rid)
        # by position, so identical repeated trades are all returned
        start_i = next((i for i, t in enumerate(self.trades) if t[0] >= startDateTime), len(self.trades))
        end_i = min(start_i + numberOfTicks, len(self.trades))
        if end_i > start_i:
            last = self.trades[end_i - 1][0].replace(microsecond=0)
            while end_i < len(self.trades) and self.trades[end_i][0] < last + timedelta(seconds=1):
                end_i += 1
        return [Tick(t[0], t[1], t[2], "CME", "", Attr(False, False)) for t in self.trades[start_i:end_i]]

    async def reqHistoricalDataAsync(self, contract, endDateTime, durationStr, barSizeSetting,
                                     whatToShow, useRTH, formatDate, timeout=0, **kw):
        rid = self.client.getReqId()
        self.n_bar_calls += 1
        action = self.bars_script.get(self.n_bar_calls)
        if action:
            return await self._fail(action, rid)
        bars = {}
        for t in self.trades:
            if endDateTime - timedelta(days=1) <= t[0] < endDateTime:
                m = t[0].replace(second=0, microsecond=0)
                o, h, l, c, v = bars.get(m, (t[1], t[1], t[1], t[1], 0))
                bars[m] = (o, max(h, t[1]), min(l, t[1]), t[1], v + t[2])
        return [Bar(m, *x) for m, x in sorted(bars.items())]
