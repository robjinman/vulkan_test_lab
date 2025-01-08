#pragma once

#include <string>
#include <memory>

class Logger
{
  public:
    virtual void debug(const std::string& msg, bool newline = true) = 0;
    virtual void info(const std::string& msg, bool newline = true) = 0;
    virtual void warn(const std::string& msg, bool newline = true) = 0;
    virtual void error(const std::string& msg, bool newline = true) = 0;

    virtual ~Logger() {}
};

using LoggerPtr = std::unique_ptr<Logger>;

LoggerPtr createLogger(std::ostream& errorStream, std::ostream& warningStream,
  std::ostream& infoStream, std::ostream& debugStream);

#ifndef NDEBUG
  #define DBG_LOG(logger, msg) logger.debug(msg);
#else
  #define DBG_LOG(logger, msg)
#endif
