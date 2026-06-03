module;

#include "TaskScheduler.h"

export module enki;

export namespace enki {
    using enki::ICompletable;
    using enki::IPinnedTask;
    using enki::ITaskSet;
    using enki::TaskSet;
    using enki::TaskScheduler;
    using enki::TaskPipe;
    using enki::PinnedTaskList;
}
