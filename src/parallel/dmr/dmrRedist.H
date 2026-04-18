/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2026 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

Description
    Malleable checkpoint/restart helpers for DMR-enabled solvers.

    dmrCheckpoint() — called by OLD processes on a DMR reconfiguration.
        Writes all fields, reconstructs to a serial case on disk, and
        removes the processor time directories.  Uses the libparallel
        reconstruction API instead of spawning a reconstructPar child
        process.

    dmrRestart() — called by NEW processes at startup.
        Updates decomposeParDict (new processor count, scotch decomposer)
        and controlDict (startFrom latestTime), then decomposes the serial
        case for the new process count using the libparallel decomposition
        API instead of spawning a decomposePar child process.

    Both functions are implemented in dmrRedist.C and compiled into
    libfoamDmr, which is linked by foamRun and foamMultiRun.

    Two small C-only helpers are defined inline here so they are usable
    before OpenFOAM is fully initialised (before setRootCase.H):

        dmrFindLatestTime()  — scans the case directory for the latest
                               numeric time directory.
        dmrReplaceLine()     — in-place line replacement in a text file;
                               replaces the sed calls in dmrRestart().

\*---------------------------------------------------------------------------*/

#ifndef dmrRedist_H
#define dmrRedist_H

#include "Time.H"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <dirent.h>

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

// ---- C-only helpers (safe before OpenFOAM initialisation) ----------------

// Scan casePath for the latest numeric directory (e.g. "0.5", "1", "2.3").
// Returns "0" if none found.  Uses only POSIX dirent — no Foam types.
inline std::string dmrFindLatestTime(const std::string& casePath)
{
    DIR* dir = opendir(casePath.c_str());
    if (!dir) return "0";

    double maxTime = -1;
    std::string maxTimeName;
    struct dirent* entry;

    while ((entry = readdir(dir)) != nullptr)
    {
        char* endptr;
        double val = strtod(entry->d_name, &endptr);

        if (*endptr == '\0' && endptr != entry->d_name && val > maxTime)
        {
            maxTime = val;
            maxTimeName = entry->d_name;
        }
    }

    closedir(dir);
    return maxTimeName.empty() ? "0" : maxTimeName;
}


// Replace the first line whose trimmed content starts with `key` with
// `newLine\n`.  If the key is absent the line is appended.
// Returns true on success, false on I/O error.
inline bool dmrReplaceLine
(
    const std::string& filePath,
    const std::string& key,
    const std::string& newLine
)
{
    FILE* fp = std::fopen(filePath.c_str(), "r");
    if (!fp)
    {
        std::fprintf(stderr,
            "DMR: cannot open '%s' for reading\n", filePath.c_str());
        return false;
    }

    std::string content;
    char buf[1024];
    while (std::fgets(buf, sizeof(buf), fp))
        content += buf;
    std::fclose(fp);

    bool found = false;
    std::string result;
    result.reserve(content.size() + newLine.size());

    std::size_t pos = 0;
    while (pos < content.size())
    {
        const std::size_t eol = content.find('\n', pos);
        const std::size_t end =
            (eol == std::string::npos) ? content.size() : eol + 1;

        const std::string line = content.substr(pos, end - pos);

        if (!found)
        {
            const std::size_t ws = line.find_first_not_of(" \t");
            if (ws != std::string::npos &&
                line.compare(ws, key.size(), key) == 0)
            {
                result += newLine + '\n';
                found = true;
                pos = end;
                continue;
            }
        }

        result += line;
        pos = end;
    }

    if (!found)
    {
        if (!result.empty() && result.back() != '\n')
            result += '\n';
        result += newLine + '\n';
    }

    fp = std::fopen(filePath.c_str(), "w");
    if (!fp)
    {
        std::fprintf(stderr,
            "DMR: cannot open '%s' for writing\n", filePath.c_str());
        return false;
    }
    std::fwrite(result.c_str(), 1, result.size(), fp);
    std::fclose(fp);
    return true;
}


// ---- Main functions (implemented in dmrRedist.C) -------------------------

namespace Foam
{

// Called by OLD processes on a DMR reconfiguration.
// Writes all fields, reconstructs to serial, removes processor time dirs.
// runTime     : the solver's current Time object (already initialised).
// allRegions  : true for foamMultiRun multi-region cases.
void dmrCheckpoint(Time& runTime, bool allRegions = false);

// Phase 1 — called by NEW processes inside DMR_AUTO (before dmr_reconfigure).
// Only updates decomposeParDict and controlDict; sets the dmrRestartPending_
// flag.  No decomposition is done here — DMR_INTERCOMM is still live.
// casePath   : absolute path to the case directory (std::string).
// allRegions : true for foamMultiRun multi-region cases.
void dmrRestart(const std::string& casePath, bool allRegions = false);

// Phase 2 — called by NEW processes AFTER DMR_AUTO (after dmr_reconfigure has
// freed DMR_INTERCOMM).  Decomposes the serial case for the new process count.
// Single-region: uses the in-process OpenFOAM decomposition API.
// Multi-region : falls back to a decomposePar subprocess.
// No-op if dmrRestartPending_ is false (fresh start, no reconfiguration).
// casePath   : absolute path to the case directory (std::string).
// allRegions : true for foamMultiRun multi-region cases.
void dmrDecomposeInProcess
(
    const std::string& casePath,
    bool allRegions = false
);

} // End namespace Foam


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
