#include <cctype>
#include <string>
#include <string_view>

#include <gtest/gtest.h>

#include "lunar_path_planner/v3/crypto/sha256.hpp"

namespace lpp = lunar::planning::v3;

TEST(Sha256, MatchesEmptyStringVector) {
  const auto result = lpp::Sha256Hex("");

  ASSERT_TRUE(lpp::IsOk(result));
  EXPECT_EQ(
      std::get<lpp::Sha256Digest>(result),
      "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
}

TEST(Sha256, MatchesAbcVector) {
  const auto result = lpp::Sha256Hex("abc");

  ASSERT_TRUE(lpp::IsOk(result));
  EXPECT_EQ(
      std::get<lpp::Sha256Digest>(result),
      "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
}

TEST(Sha256, ReturnsExactly64LowercaseHexCharacters) {
  const auto result = lpp::Sha256Hex("path-planner-v3");

  ASSERT_TRUE(lpp::IsOk(result));
  const auto& digest = std::get<lpp::Sha256Digest>(result);
  ASSERT_EQ(digest.size(), 64U);
  for (const char character : digest) {
    EXPECT_TRUE((character >= '0' && character <= '9') ||
                (character >= 'a' && character <= 'f'));
  }
}
