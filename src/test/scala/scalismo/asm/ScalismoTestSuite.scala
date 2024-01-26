package scalismo.asm

import org.scalatest.funspec.AnyFunSpec
import org.scalatest.matchers.should.Matchers
import scalismo.vtk.scalismo

class ScalismoTestSuite extends AnyFunSpec with Matchers {
  scalismo.initialize()
}
